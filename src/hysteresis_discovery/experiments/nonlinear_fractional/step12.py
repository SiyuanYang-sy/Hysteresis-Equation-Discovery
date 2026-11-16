"""STEP12: nonlinear fractional."""

from ...runtime import RunConfig, experiment_session


def run(config: RunConfig) -> None:
    """Run the experiment with shared paths and runtime settings."""
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from torchdiffeq import odeint
    import numpy as np
    import time
    from ...numerics import rmse, diff_series, lerp_series
    from ...networks import XDotDotNet

    with experiment_session(config, "nonlinear_fractional", "step12") as session:
        path = str(session.output_dir) + "/"
        device = session.device
        time0 = time.time()

        fs = 100
        dt = 1.0 / fs
        T = 10.0
        t = torch.arange(0.0, T, dt, device=device)
        u = torch.sin(2 * t).to(device)

        t0 = t[0]

        m_true = 1.0
        c_true = 0.8
        k_true = 0.5
        alpha_true = 1.0
        A_true = 4.0
        beta_true = 5.0
        gamma_true = -4.0
        n_true = 1.5

        def true_bouc_wen_nonlinear(t_local, state):
            x, x_dot, z = state
            f = lerp_series(u, t_local, t0, dt)
            dx = x_dot
            dv = (f - c_true * x_dot - k_true * x**3 - alpha_true * z) / m_true
            # dv = (f - c_true * x_dot - k_true * x - alpha_true * z) / m_true
            dz = (
                A_true * x_dot
                - beta_true * torch.abs(x_dot) * torch.abs(z) ** (n_true - 1) * z
                - gamma_true * x_dot * torch.abs(z) ** n_true
            )
            return torch.stack([dx, dv, dz])

        x0 = torch.tensor([0.0, 0.0, 0.0], device=device)
        with torch.no_grad():
            sol_true = odeint(true_bouc_wen_nonlinear, x0, t, method="rk4")

        x_solver, x_dot_solver, z_solver = (
            sol_true[:, 0],
            sol_true[:, 1],
            sol_true[:, 2],
        )

        x_true = x_solver
        x_dot_true = x_dot_solver
        z_true = z_solver
        x_dotdot_true = (
            u - c_true * x_dot_true - k_true * x_true**3 - alpha_true * z_true
        ) / m_true
        # x_dotdot_true = (u - c_true * x_dot_true - k_true * x_true - alpha_true * z_true) / m_true
        z_dot_true = (
            A_true * x_dot_true
            - beta_true
            * torch.abs(x_dot_true)
            * torch.abs(z_true) ** (n_true - 1)
            * z_true
            - gamma_true * x_dot_true * torch.abs(z_true) ** n_true
        )

        plt.figure()
        plt.plot(t.cpu(), u.cpu(), color="gray", linewidth=2)
        plt.xlabel(r"Time step $t$")
        plt.ylabel(r"External force $u$")
        plt.title(r"Solution of Nonlinear Bouc-Wen System")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.plot(t.cpu(), x_true.cpu(), color="gray", linewidth=2)
        plt.xlabel(r"Time step $t$")
        plt.ylabel(r"Displacement $x$")
        plt.title(r"Solution of Nonlinear Bouc-Wen System")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.plot(t.cpu(), x_dot_true.cpu(), color="gray", linewidth=2)
        plt.xlabel(r"Time step $t$")
        plt.ylabel(r"Velocity $\dot{x}$")
        plt.title(r"Solution of Nonlinear Bouc-Wen System")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.plot(t.cpu(), z_true.cpu(), color="gray", linewidth=2)
        plt.xlabel(r"Time step $t$")
        plt.ylabel(r"Hysteretic force $z$")
        plt.title(r"Solution of Nonlinear Bouc-Wen System")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.plot(t.cpu(), z_dot_true.cpu(), color="gray", linewidth=2)
        plt.xlabel(r"Time step $t$")
        plt.ylabel(r"Hysteretic force derivative $\dot{z}$")
        plt.title(r"Solution of Nonlinear Bouc-Wen System")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.plot(x_true.cpu(), z_true.cpu(), color="gray", linewidth=2)
        plt.xlabel(r"Displacement $x$")
        plt.ylabel(r"Hysteretic force $z$")
        plt.title(r"$x$-$z$ Hysteresis loop learned from solver-based model")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Step 1: Assume z = 0 and learn [m, c, k] under the linearized model: m*x'' + c*x' + k*x = u
        # Original model: m*x'' + c*x' + k*x + alpha*z = u
        print(
            "###################################### Step 1 ######################################"
        )
        start_index = 2 * fs
        end_index = 3 * fs
        data_t_LSM = t[start_index:end_index]
        data_u_LSM = u[start_index:end_index]

        # ---- Linear Bouc-Wen ----
        t0 = data_t_LSM[0]

        def true_bouc_wen(t_local, state):
            x, x_dot = state
            f = lerp_series(data_u_LSM, t_local, t0, dt)
            dx = x_dot
            dv = (f - c_true * x_dot - k_true * x) / m_true
            return torch.stack([dx, dv])

        x0 = torch.tensor([x_true[start_index], x_dot_true[start_index]], device=device)
        with torch.no_grad():
            sol_true = odeint(true_bouc_wen, x0, data_t_LSM, method="rk4")
        x_solver, x_dot_solver = sol_true[:, 0], sol_true[:, 1]
        x_dotdot_solver = (
            data_u_LSM - c_true * x_dot_solver - k_true * x_solver
        ) / m_true

        data_x_dotdot_LSM, data_x_dot_LSM, data_x_LSM = (
            x_dotdot_solver.detach().cpu(),
            x_dot_solver.detach().cpu(),
            x_solver.detach().cpu(),
        )

        X_matrix = np.vstack((data_x_dotdot_LSM, data_x_dot_LSM, data_x_LSM)).T
        Y_vector = data_u_LSM.detach().cpu()

        X_matrix = np.ascontiguousarray(
            X_matrix.numpy() if hasattr(X_matrix, "numpy") else X_matrix,
            dtype=np.float64,
        )
        Y_vector = np.ascontiguousarray(
            Y_vector.numpy() if hasattr(Y_vector, "numpy") else Y_vector,
            dtype=np.float64,
        )

        print("X_matrix finite:", np.isfinite(X_matrix).all())
        print("Y_vector finite:", np.isfinite(Y_vector).all())
        print("X_matrix shape:", X_matrix.shape, "Y_vector shape:", Y_vector.shape)

        para_mck = np.linalg.lstsq(X_matrix, Y_vector, rcond=None)[0]

        m_para_LSM, c_para_LSM, k_para_LSM = para_mck
        print(
            f"LSM Learned Parameters [m, c, k]: m_para={m_para_LSM:.4f}, c_para={c_para_LSM:.4f}, k_para={k_para_LSM:.4f}"
        )

        # Step 2: Learn z
        print(
            "###################################### Step 2 ######################################"
        )

        t0 = t[0]
        m0 = m_para_LSM
        c0 = c_para_LSM
        k0 = k_para_LSM
        alpha0 = np.random.normal(loc=1.0, scale=np.sqrt(0.01))
        print(f"m0={m0:.4f}, c0={c0:.4f}, k0={k0:.4f}, alpha0={alpha0:.4f}")
        z0 = (u - m0 * x_dotdot_true - c0 * x_dot_true - k0 * x_true) / alpha0

        class LearnableZBoucWen(nn.Module):
            def __init__(self, hidden_dim=128):
                super().__init__()
                self.m = nn.Parameter(torch.tensor(m0, device=device))
                self.c = nn.Parameter(torch.tensor(c0, device=device))
                self.k = nn.Parameter(torch.tensor(k0, device=device))
                self.alpha = nn.Parameter(torch.tensor(alpha0, device=device))

                self.x_dotdot_net = XDotDotNet(hidden_dim=hidden_dim)

                self.z_param = nn.Parameter(z0)
                self.z_grad = None

            def update_z_grad(self, dt):
                self.z_grad = diff_series(self.z_param, dt)

            def forward(self, t, state):
                x, x_dot, z = state
                f = lerp_series(u, t, t0, dt)
                z = lerp_series(self.z_param, t, t0, dt)
                dx = x_dot
                dx_dot = self.x_dotdot_net(
                    x.unsqueeze(0), x_dot.unsqueeze(0), z.unsqueeze(0), f.unsqueeze(0)
                )[0]
                assert (
                    self.z_grad is not None
                ), "call model.update_z_grad(dt) before odeint"
                dz = lerp_series(self.z_grad, t, t0, dt)

                return torch.stack([dx, dx_dot, dz])

        model = LearnableZBoucWen().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3e-4, betas=(0.9, 0.98), weight_decay=1e-4
        )
        epochs = config.epochs
        loss_history = []

        ep_str = f" (Epoch {epochs})"
        for epoch in range(epochs):
            model.update_z_grad(dt)
            x0 = torch.stack([x_true[0], x_dot_true[0], model.z_param[0].detach()]).to(
                device
            )
            pred = odeint(model, x0, t, method="rk4")
            x_pred, x_dot_pred, z_pred = pred[:, 0], pred[:, 1], pred[:, 2]
            z_learned = model.z_param
            z_dot_learned = diff_series(z_learned, dt)
            sta_ind_loss = 2 * fs
            end_ind_loss = 6 * fs
            t_loss = t[sta_ind_loss:end_ind_loss]
            x_pred_loss, x_dot_pred_loss, z_pred_loss = (
                x_pred[sta_ind_loss:end_ind_loss],
                x_dot_pred[sta_ind_loss:end_ind_loss],
                z_pred[sta_ind_loss:end_ind_loss],
            )
            x_true_loss, x_dot_true_loss, z_true_loss, z_dot_true_loss = (
                x_true[sta_ind_loss:end_ind_loss],
                x_dot_true[sta_ind_loss:end_ind_loss],
                z_true[sta_ind_loss:end_ind_loss],
                z_dot_true[sta_ind_loss:end_ind_loss],
            )
            z_param_loss, z_dot_loss = (
                z_learned[sta_ind_loss:end_ind_loss],
                z_dot_learned[sta_ind_loss:end_ind_loss],
            )
            x_dotdot_pred = model.x_dotdot_net(x_pred, x_dot_pred, z_learned, u)
            x_dotdot_pred_loss, x_dotdot_true_loss = (
                x_dotdot_pred[sta_ind_loss:end_ind_loss],
                x_dotdot_true[sta_ind_loss:end_ind_loss],
            )

            RMSE_x = rmse(x_pred_loss, x_true_loss)
            NRMSE_x = RMSE_x / (x_true_loss.max() - x_true_loss.min() + 1e-8)
            RMSE_x_dot = rmse(x_dot_pred_loss, x_dot_true_loss)
            NRMSE_x_dot = RMSE_x_dot / (
                x_dot_true_loss.max() - x_dot_true_loss.min() + 1e-8
            )
            RMSE_x_dotdot = rmse(x_dotdot_pred_loss, x_dotdot_true_loss)
            NRMSE_x_dotdot = RMSE_x_dotdot / (
                x_dotdot_true_loss.max() - x_dotdot_true_loss.min() + 1e-8
            )

            RMSE_z = rmse(z_param_loss, z_pred_loss)
            NRMSE_z = RMSE_z / (z_pred_loss.max() - z_pred_loss.min() + 1e-8)

            lam_z = 1e-3
            z_second = z_learned[2:] - 2 * z_learned[1:-1] + z_learned[:-2]
            smooth_penalty = (z_second**2).mean()

            loss = NRMSE_x + NRMSE_x_dot + lam_z * smooth_penalty

            # Compute the NRMSE of z (the true value is, of course, unknown in practice)
            RMSE_z_true = rmse(z_param_loss, z_true_loss)
            NRMSE_z_true = RMSE_z_true / (z_true_loss.max() - z_true_loss.min() + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_history.append(loss.item())
            if (epoch == 0) or ((epoch + 1) % 5 == 0):
                print(
                    f"Epoch {epoch+1} | NRMSE Loss: {loss.item():.4e} | "
                    f"NRMSE_x: {NRMSE_x.item():.4f} | NRMSE_x_dot: {NRMSE_x_dot.item():.4f} | NRMSE_z: {NRMSE_z.item():.4f} | "
                    f"NRMSE_z_true: {NRMSE_z_true.item():.4f} | NRMSE_x_dotdot: {NRMSE_x_dotdot.item():.4f} | "
                    f"m: {model.m.item():.4f} (true: {m_true:.1f}) | c: {model.c.item():.4f} (true: {c_true:.1f}) | "
                    f"k: {model.k.item():.4f} (true: {k_true:.1f}) | alpha: {model.alpha.item():.4f} (true: {alpha_true:.1f})"
                )
            if session.checkpoint(
                epoch, 10, include_first=True, interval=config.plot_interval
            ):
                ep_str = f" (Epoch {epoch + 1})"

                plt.figure()
                plt.plot(loss_history)
                plt.title(r"Training Loss" + ep_str)
                plt.xlabel(r"Epoch")
                plt.ylabel(r"Loss Value")
                plt.tight_layout()
                plt.savefig(path + f"Training_Loss_Epoch_{epoch + 1}.png", dpi=600)
                plt.close()

                plt.figure()
                plt.plot(
                    t_loss.detach().cpu(),
                    x_true_loss.detach().cpu(),
                    label=r"True $x(t)$",
                )
                plt.plot(
                    t_loss.detach().cpu(),
                    x_pred_loss.detach().cpu(),
                    "--",
                    label=r"Predicted $x(t)$",
                )
                plt.title(r"$x(t)$ Comparison" + ep_str)
                plt.xlabel(r"Time step")
                plt.ylabel(r"$x$ value")
                plt.legend()
                plt.tight_layout()
                plt.savefig(path + f"x_t_Comparison_Epoch_{epoch + 1}.png", dpi=600)
                plt.close()

                plt.figure()
                plt.plot(
                    t_loss.detach().cpu(),
                    x_dot_true_loss.detach().cpu(),
                    label=r"True $\dot{x}(t)$",
                )
                plt.plot(
                    t_loss.detach().cpu(),
                    x_dot_pred_loss.detach().cpu(),
                    "--",
                    label=r"Predicted $\dot{x}(t)$",
                )
                plt.title(r"$\dot{x}(t)$ Comparison" + ep_str)
                plt.xlabel(r"Time step")
                plt.ylabel(r"$\dot{x}$ value")
                plt.legend()
                plt.tight_layout()
                plt.savefig(path + f"x_dot_t_Comparison_Epoch_{epoch + 1}.png", dpi=600)
                plt.close()

                plt.figure()
                plt.plot(
                    t_loss.detach().cpu(),
                    x_dotdot_true_loss.detach().cpu(),
                    label=r"True $\ddot{x}(t)$",
                )
                plt.plot(
                    t_loss.detach().cpu(),
                    x_dotdot_pred_loss.detach().cpu(),
                    "--",
                    label=r"Predicted $\ddot{x}(t)$",
                )
                plt.title(r"$\ddot{x}(t)$ Comparison" + ep_str)
                plt.xlabel(r"Time step")
                plt.ylabel(r"$\ddot{x}$ value")
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    path + f"x_dotdot_t_Comparison_Epoch_{epoch + 1}.png", dpi=600
                )
                plt.close()

                plt.figure()
                plt.plot(
                    t_loss.detach().cpu(),
                    z_true_loss.detach().cpu(),
                    label=r"True $z(t)$",
                )
                plt.plot(
                    t_loss.detach().cpu(),
                    z_param_loss.detach().cpu(),
                    "--",
                    label=r"Learned $z(t)$",
                )
                plt.title(r"Learned $z(t)$ Trajectory" + ep_str)
                plt.xlabel(r"Time step")
                plt.ylabel(r"$z$ value")
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    path + f"Learned_z_t_Trajectory_Epoch_{epoch + 1}.png", dpi=600
                )
                plt.close()

                plt.figure()
                plt.plot(
                    t_loss.detach().cpu(),
                    z_dot_true_loss.detach().cpu(),
                    label=r"True $\dot{z}(t)$",
                )
                plt.plot(
                    t_loss.detach().cpu(),
                    z_dot_loss.detach().cpu(),
                    "--",
                    label=r"Learned $\dot{z}(t)$",
                )
                plt.title(r"Learned $\dot{z}(t)$ Trajectory" + ep_str)
                plt.xlabel(r"Time step")
                plt.ylabel(r"$\dot{z}$ value")
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    path + f"Learned_z_dot_t_Trajectory_Epoch_{epoch + 1}.png", dpi=600
                )
                plt.close()

                plt.figure()
                plt.plot(
                    x_true_loss.detach().cpu(),
                    z_true_loss.detach().cpu(),
                    label=r"True $x$-$z$",
                )
                plt.plot(
                    x_pred_loss.detach().cpu(),
                    z_param_loss.detach().cpu(),
                    "--",
                    label=r"Learned $x$-$z$",
                )
                plt.title(
                    r"$x$-$z$ Hysteresis loop learned from solver-based model" + ep_str
                )
                plt.xlabel(r"$x$ value")
                plt.ylabel(r"$z$ value")
                plt.legend()
                plt.tight_layout()
                plt.savefig(path + f"Hysteresis_loop_Epoch_{epoch + 1}.png", dpi=600)
                plt.close()

                np.savez(
                    path + f"data_Epoch_{epoch + 1}.npz",
                    t=t.cpu().numpy(),
                    u=u.cpu().numpy(),
                    x0=x0.cpu().numpy(),
                    x_true=x_true.cpu().numpy(),
                    x_pred=x_pred.detach().cpu().numpy(),
                    x_dot_true=x_dot_true.cpu().numpy(),
                    x_dot_pred=x_dot_pred.detach().cpu().numpy(),
                    x_dotdot_true=x_dotdot_true.cpu().numpy(),
                    x_dotdot_pred=x_dotdot_pred.detach().cpu().numpy(),
                    z_true=z_true.cpu().numpy(),
                    z_learned=z_learned.detach().cpu().numpy(),
                    z_dot_true=z_dot_true.cpu().numpy(),
                    z_dot_learned=z_dot_learned.detach().cpu().numpy(),
                    m=float(model.m),
                    c=float(model.c),
                    k=float(model.k),
                    alpha=float(model.alpha),
                    loss_history=loss_history,
                )

        with torch.no_grad():
            model.update_z_grad(dt)
            x0 = torch.tensor(
                [x_true[0], x_dot_true[0], model.z_param[0]], device=device
            )
            pred = odeint(model, x0, t, method="rk4")
            x_pred, x_dot_pred, z_pred = pred[:, 0], pred[:, 1], pred[:, 2]
            z_learned = model.z_param
            z_dot_learned = diff_series(z_learned, dt)
            sta_ind_loss = 0
            end_ind_loss = -2
            t_loss = t[sta_ind_loss:end_ind_loss]
            x_pred_loss, x_dot_pred_loss, z_pred_loss = (
                x_pred[sta_ind_loss:end_ind_loss],
                x_dot_pred[sta_ind_loss:end_ind_loss],
                z_pred[sta_ind_loss:end_ind_loss],
            )
            x_true_loss, x_dot_true_loss, z_true_loss, z_dot_true_loss = (
                x_true[sta_ind_loss:end_ind_loss],
                x_dot_true[sta_ind_loss:end_ind_loss],
                z_true[sta_ind_loss:end_ind_loss],
                z_dot_true[sta_ind_loss:end_ind_loss],
            )
            z_param_loss, z_dot_loss = (
                z_learned[sta_ind_loss:end_ind_loss],
                z_dot_learned[sta_ind_loss:end_ind_loss],
            )
            x_dotdot_pred = model.x_dotdot_net(x_pred, x_dot_pred, z_learned, u)
            x_dotdot_pred_loss, x_dotdot_true_loss = (
                x_dotdot_pred[sta_ind_loss:end_ind_loss],
                x_dotdot_true[sta_ind_loss:end_ind_loss],
            )
        print(f"Total time: {time.time()-time0: .4f} s")

        plt.figure()
        plt.plot(loss_history)
        plt.title(r"Training Loss" + ep_str)
        plt.xlabel(r"Epoch")
        plt.ylabel(r"Loss Value")
        plt.tight_layout()
        plt.savefig(path + f"Training_Loss_Epoch_{epoch + 1}.png", dpi=600)
        plt.close()

        plt.figure()
        plt.plot(
            t_loss.detach().cpu(), x_true_loss.detach().cpu(), label=r"True $x(t)$"
        )
        plt.plot(
            t_loss.detach().cpu(),
            x_pred_loss.detach().cpu(),
            "--",
            label=r"Predicted $x(t)$",
        )
        plt.title(r"$x(t)$ Comparison" + ep_str)
        plt.xlabel(r"Time step")
        plt.ylabel(r"$x$ value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path + f"x_t_Comparison_Epoch_{epoch + 1}.png", dpi=600)
        plt.close()

        plt.figure()
        plt.plot(
            t_loss.detach().cpu(),
            x_dot_true_loss.detach().cpu(),
            label=r"True $\dot{x}(t)$",
        )
        plt.plot(
            t_loss.detach().cpu(),
            x_dot_pred_loss.detach().cpu(),
            "--",
            label=r"Predicted $\dot{x}(t)$",
        )
        plt.title(r"$\dot{x}(t)$ Comparison" + ep_str)
        plt.xlabel(r"Time step")
        plt.ylabel(r"$\dot{x}$ value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path + f"x_dot_t_Comparison_Epoch_{epoch + 1}.png", dpi=600)
        plt.close()

        plt.figure()
        plt.plot(
            t_loss.detach().cpu(),
            x_dotdot_true_loss.detach().cpu(),
            label=r"True $\ddot{x}(t)$",
        )
        plt.plot(
            t_loss.detach().cpu(),
            x_dotdot_pred_loss.detach().cpu(),
            "--",
            label=r"Predicted $\ddot{x}(t)$",
        )
        plt.title(r"$\ddot{x}(t)$ Comparison" + ep_str)
        plt.xlabel(r"Time step")
        plt.ylabel(r"$\ddot{x}$ value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path + f"x_dotdot_t_Comparison_Epoch_{epoch + 1}.png", dpi=600)
        plt.close()

        plt.figure()
        plt.plot(
            t_loss.detach().cpu(), z_true_loss.detach().cpu(), label=r"True $z(t)$"
        )
        plt.plot(
            t_loss.detach().cpu(),
            z_param_loss.detach().cpu(),
            "--",
            label=r"Learned $z(t)$",
        )
        plt.title(r"Learned $z(t)$ Trajectory" + ep_str)
        plt.xlabel(r"Time step")
        plt.ylabel(r"$z$ value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path + f"Learned_z_t_Trajectory_Epoch_{epoch + 1}.png", dpi=600)
        plt.close()

        plt.figure()
        plt.plot(
            t_loss.detach().cpu(),
            z_dot_true_loss.detach().cpu(),
            label=r"True $\dot{z}(t)$",
        )
        plt.plot(
            t_loss.detach().cpu(),
            z_dot_loss.detach().cpu(),
            "--",
            label=r"Learned $\dot{z}(t)$",
        )
        plt.title(r"Learned $\dot{z}(t)$ Trajectory" + ep_str)
        plt.xlabel(r"Time step")
        plt.ylabel(r"$\dot{z}$ value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path + f"Learned_z_dot_t_Trajectory_Epoch_{epoch + 1}.png", dpi=600)
        plt.close()

        plt.figure()
        plt.plot(
            x_true_loss.detach().cpu(),
            z_true_loss.detach().cpu(),
            label=r"True $x$-$z$",
        )
        plt.plot(
            x_pred_loss.detach().cpu(),
            z_param_loss.detach().cpu(),
            "--",
            label=r"Learned $x$-$z$",
        )
        plt.title(r"$x$-$z$ Hysteresis loop learned from solver-based model" + ep_str)
        plt.xlabel(r"$x$ value")
        plt.ylabel(r"$z$ value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path + f"Hysteresis_loop_Epoch_{epoch + 1}.png", dpi=600)
        plt.close()

        np.savez(
            path + f"data_Final.npz",
            t=t.cpu().numpy(),
            u=u.cpu().numpy(),
            x0=x0.cpu().numpy(),
            x_true=x_true.cpu().numpy(),
            x_pred=x_pred.detach().cpu().numpy(),
            x_dot_true=x_dot_true.cpu().numpy(),
            x_dot_pred=x_dot_pred.detach().cpu().numpy(),
            x_dotdot_true=x_dotdot_true.cpu().numpy(),
            x_dotdot_pred=x_dotdot_pred.detach().cpu().numpy(),
            z_true=z_true.cpu().numpy(),
            z_learned=z_learned.detach().cpu().numpy(),
            z_dot_true=z_dot_true.cpu().numpy(),
            z_dot_learned=z_dot_learned.detach().cpu().numpy(),
            m=float(model.m),
            c=float(model.c),
            k=float(model.k),
            alpha=float(model.alpha),
            loss_history=loss_history,
        )

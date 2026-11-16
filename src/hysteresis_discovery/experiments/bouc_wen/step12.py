"""STEP12: bouc wen."""

from ...runtime import RunConfig, experiment_session


def run(config: RunConfig) -> None:
    """Run the experiment with shared paths and runtime settings."""
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from torchdiffeq import odeint
    import numpy as np
    import scipy.io
    import time
    from ...numerics import rmse, diff_series, lerp_series
    from ...networks import XDotDotNet

    with experiment_session(config, "bouc_wen", "step12") as session:
        path = str(session.output_dir) + "/"
        device = session.device
        time0 = time.time()

        # ----- Data Loading -----
        fs = 750
        dt = 1 / fs
        duration = 204
        t_raw = np.linspace(
            0, duration, fs * duration
        )  # Time axis: T = duration, N = fs * duration = 153000
        u_raw = scipy.io.loadmat(session.data_dir / "uval_sinesweep.mat")[
            "uval_sinesweep"
        ].squeeze()
        x_raw = scipy.io.loadmat(session.data_dir / "yval_sinesweep.mat")[
            "yval_sinesweep"
        ].squeeze()
        x_dot_raw = np.gradient(x_raw, t_raw)
        x_dotdot_raw = np.gradient(x_dot_raw, t_raw)

        start_point = 0
        end_point = 1000
        t = torch.tensor(t_raw[start_point:end_point], device=device)
        u = torch.tensor(u_raw[start_point:end_point], device=device)
        x_true = torch.tensor(x_raw[start_point:end_point], device=device)
        x_dot_true = torch.tensor(x_dot_raw[start_point:end_point], device=device)
        x_dotdot_true = torch.tensor(x_dotdot_raw[start_point:end_point], device=device)

        m, c, k, alpha = 2.0, 10.0, 50000.0, 1.0
        A, beta, gamma, n = 50000.0, 800, -1100, 1.0

        t0 = t[0]

        def true_bouc_wen(t_local, state):
            x, x_dot, z = state
            idx = int(torch.clamp(((t_local - t0) / dt).round(), 0, len(u) - 1).item())
            f = u[idx]
            dx = x_dot
            dv = (f - c * x_dot - k * x - alpha * z) / m
            dz = (
                A * x_dot
                - beta * torch.abs(x_dot) * torch.abs(z) ** (n - 1) * z
                - gamma * x_dot * torch.abs(z) ** n
            )
            return torch.stack([dx, dv, dz])

        x0 = torch.tensor([0.0, 0.0, 0.0], device=device)
        with torch.no_grad():
            sol_true = odeint(true_bouc_wen, x0, t, method="rk4")
        # RK4 solver
        x_solver, x_dot_solver, z_solver = (
            sol_true[:, 0],
            sol_true[:, 1],
            sol_true[:, 2],
        )
        x_true, x_dot_true = x_solver, x_dot_solver
        z_true = z_solver
        x_dotdot_true = (u - c * x_dot_true - k * x_true - alpha * z_true) / m
        z_dot_true = (
            A * x_dot_true
            - beta * torch.abs(x_dot_true) * torch.abs(z_true) ** (n - 1) * z_true
            - gamma * x_dot_true * torch.abs(z_true) ** n
        )

        plt.figure()
        plt.plot(t.cpu(), x_true.cpu(), label=r"True $x(t)$")
        plt.plot(t.cpu(), x_solver.cpu(), "--", label=r"Solver $x(t)$")
        plt.title(r"Benchmark $x(t)$ vs Solver $x(t)$")
        plt.xlabel(r"Epoch")
        plt.ylabel(r"Loss Value")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(t.cpu(), x_dot_true.cpu(), label=r"True $\dot{x}(t)$")
        plt.plot(t.cpu(), x_dot_solver.cpu(), "--", label=r"Solver $\dot{x}(t)$")
        plt.title(r"Benchmark $\dot{x}(t)$ vs Solver $\dot{x}(t)$")
        plt.xlabel(r"Epoch")
        plt.ylabel(r"Loss Value")
        plt.legend()
        plt.show()

        start_point, end_point = 500, 800
        t = t[start_point:end_point]
        u = u[start_point:end_point]
        x_true = x_true[start_point:end_point]
        x_dot_true = x_dot_true[start_point:end_point]
        x_dotdot_true = x_dotdot_true[start_point:end_point]
        z_true = z_true[start_point:end_point]

        t0 = t[0]

        def true_bouc_wen(t_local, state):
            x, x_dot, z = state
            idx = int(torch.clamp(((t_local - t0) / dt).round(), 0, len(u) - 1).item())
            f = u[idx]
            dx = x_dot
            dv = (f - c * x_dot - k * x - alpha * z) / m
            dz = (
                A * x_dot
                - beta * torch.abs(x_dot) * torch.abs(z) ** (n - 1) * z
                - gamma * x_dot * torch.abs(z) ** n
            )
            return torch.stack([dx, dv, dz])

        x0 = torch.tensor([x_true[0], x_dot_true[0], z_true[0]], device=device)
        with torch.no_grad():
            sol_true = odeint(true_bouc_wen, x0, t, method="rk4")
        x_solver, x_dot_solver, z_solver = (
            sol_true[:, 0],
            sol_true[:, 1],
            sol_true[:, 2],
        )
        x_true, x_dot_true = x_solver, x_dot_solver
        z_true = z_solver
        x_dotdot_true = (u - c * x_dot_true - k * x_true - alpha * z_true) / m
        z_dot_true = (
            A * x_dot_true
            - beta * torch.abs(x_dot_true) * torch.abs(z_true) ** (n - 1) * z_true
            - gamma * x_dot_true * torch.abs(z_true) ** n
        )

        plt.figure()
        plt.plot(t.cpu(), x_true.cpu(), label=r"True $x(t)$")
        plt.plot(t.cpu(), x_solver.cpu(), "--", label=r"Solver $x(t)$")
        plt.title(r"Benchmark $x(t)$ vs Solver $x(t)$")
        plt.xlabel(r"Epoch")
        plt.ylabel(r"Loss Value")
        plt.legend()
        plt.show()
        #
        plt.figure()
        plt.plot(t.cpu(), x_dot_true.cpu(), label=r"True $\dot{x}(t)$")
        plt.plot(t.cpu(), x_dot_solver.cpu(), "--", label=r"Solver $\dot{x}(t)$")
        plt.title(r"Benchmark $\dot{x}(t)$ vs Solver $\dot{x}(t)$")
        plt.xlabel(r"Epoch")
        plt.ylabel(r"Loss Value")
        plt.legend()
        plt.show()

        # Step 1: Assume z = 0 and learn [m, c, k] under the linearized model: m*x'' + c*x' + k*x = u
        # Original model: m*x'' + c*x' + k*x + alpha*z = u
        print(
            "###################################### Step 1 ######################################"
        )
        start_index = 0
        end_index = 100
        data_t_LSM = torch.tensor(t_raw[start_index:end_index], device=device)
        data_u_LSM = torch.tensor(u_raw[start_index:end_index], device=device)

        # ---- Linear Bouc-Wen ----
        t0 = data_t_LSM[0]

        def true_bouc_wen(t_local, state):
            x, x_dot = state
            idx = int(torch.clamp(((t_local - t0) / dt).round(), 0, len(u) - 1).item())
            f = data_u_LSM[idx]
            dx = x_dot
            dv = (f - c * x_dot - k * x) / m
            return torch.stack([dx, dv])

        x0 = torch.tensor([0.0, 0.0], device=device)
        with torch.no_grad():
            sol_true = odeint(true_bouc_wen, x0, data_t_LSM, method="rk4")
        x_solver, x_dot_solver = sol_true[:, 0], sol_true[:, 1]
        x_dotdot_solver = (data_u_LSM - c * x_dot_solver - k * x_solver) / m

        data_x_dotdot_LSM, data_x_dot_LSM, data_x_LSM = (
            x_dotdot_solver.detach().cpu(),
            x_dot_solver.detach().cpu(),
            x_solver.detach().cpu(),
        )
        # # Add noise if needed
        # data_x_LSM = add_noise(data_x_LSM)
        # data_x_dotdot_LSM, data_x_dot_LSM, data_x_LSM = (
        #     add_noise(data_x_dotdot_LSM), add_noise(data_x_dot_LSM), add_noise(data_x_LSM))

        X_matrix = np.vstack(
            (data_x_dotdot_LSM, data_x_dot_LSM, data_x_LSM)  # x_dotdot  # x_dot  # x
        ).T
        Y_vector = data_u_LSM.detach().cpu()

        para_mck = np.linalg.lstsq(X_matrix, Y_vector, rcond=None)[0]

        m_para_LSM, c_para_LSM, k_para_LSM = para_mck
        print(
            f"Learned Parameters [m, c, k]: m_para={m_para_LSM:.4f}, c_para={c_para_LSM:.4f}, k_para={k_para_LSM:.4f}"
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
            def __init__(self, t_len, hidden_dim=128):
                super().__init__()
                self.m = m0
                self.c = c0
                self.k = k0
                self.alpha = nn.Parameter(torch.tensor(alpha0, device=device))
                self.x_dotdot_net = XDotDotNet(hidden_dim=hidden_dim)
                self.z_param = nn.Parameter(z0)

            def update_z_grad(self, dt):
                self.z_grad = diff_series(self.z_param, dt)

            def forward(self, t, state):
                x, x_dot, z = state
                f = lerp_series(-self.m * u, t, t0, dt)
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

        model = LearnableZBoucWen(t_len=len(t)).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3e-4, betas=(0.9, 0.98), weight_decay=1e-4
        )
        epochs = config.epochs
        loss_history = []

        for epoch in range(epochs):
            model.update_z_grad(dt)
            x0 = torch.tensor(
                [x_true[0], x_dot_true[0], model.z_param[0]], device=device
            )
            pred = odeint(model, x0, t, method="rk4")
            x_pred, x_dot_pred, z_pred = pred[:, 0], pred[:, 1], pred[:, 2]
            z_learned = model.z_param
            z_dot_learned = diff_series(z_learned, dt)
            z_phy = (
                u - model.m * x_dotdot_true - model.c * x_dot_true - model.k * x_true
            ) / model.alpha
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
            z_param_loss, z_phy_loss, z_dot_loss = (
                z_learned[sta_ind_loss:end_ind_loss],
                z_phy[sta_ind_loss:end_ind_loss],
                z_dot_learned[sta_ind_loss:end_ind_loss],
            )

            # # Add noise if needed
            # x_true_loss, x_dot_true_loss = add_noise(x_true_loss), add_noise(x_dot_true_loss)

            RMSE_x = rmse(x_pred_loss, x_true_loss)
            NRMSE_x = RMSE_x / (x_true_loss.max() - x_true_loss.min() + 1e-8)
            RMSE_x_dot = rmse(x_dot_pred_loss, x_dot_true_loss)
            NRMSE_x_dot = RMSE_x_dot / (
                x_dot_true_loss.max() - x_dot_true_loss.min() + 1e-8
            )
            RMSE_z = rmse(z_param_loss, z_phy_loss)
            NRMSE_z = RMSE_z / (z_phy_loss.max() - z_phy_loss.min() + 1e-8)

            # Compute the NRMSE of z (the true value is, of course, unknown in practice)
            RMSE_z_true = rmse(z_param_loss, z_true_loss)
            NRMSE_z_true = RMSE_z_true / (z_true_loss.max() - z_true_loss.min() + 1e-8)

            loss = NRMSE_x + NRMSE_x_dot + NRMSE_z

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            if (epoch + 1) % 50 == 0:
                print(
                    f"Epoch {epoch+1} | NRMSE Loss: {loss.item():.4e} | "
                    f"NRMSE_x: {NRMSE_x.item():.4f} | NRMSE_x_dot: {NRMSE_x_dot.item():.4f} | "
                    f"NRMSE_z: {NRMSE_z.item():.4f} | NRMSE_z_true: {NRMSE_z_true.item():.4f} | "
                    f"m: {float(model.m):.4f} (true: {m:.1f}) | c: {float(model.c):.4f} (true: {c:.1f}) | "
                    f"k: {float(model.k):.4f} (true: {k:.1f}) | alpha: {model.alpha.item():.4f} (true: {alpha:.1f})"
                )

            if session.checkpoint(
                epoch, 100, include_first=False, interval=config.plot_interval
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

                np.savez(
                    path + f"data_Epoch_{epoch + 1}_sinesweep.npz",
                    t=t.cpu().numpy(),
                    u=u.cpu().numpy(),
                    x0=x0.cpu().numpy(),
                    x_true=x_true.cpu().numpy(),
                    x_pred=x_pred.detach().cpu().numpy(),
                    x_dot_true=x_dot_true.cpu().numpy(),
                    x_dot_pred=x_dot_pred.detach().cpu().numpy(),
                    x_dotdot_true=x_dotdot_true.cpu().numpy(),
                    z_true=z_true.cpu().numpy(),
                    z_learned=z_learned.detach().cpu().numpy(),
                    z_phy=z_phy.detach().cpu().numpy(),
                    z_dot_true=z_dot_true.cpu().numpy(),
                    z_dot_learned=z_dot_learned.detach().cpu().numpy(),
                    m=float(model.m),
                    c=float(model.c),
                    k=float(model.k),
                    alpha=float(model.alpha),
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
            z_phy = (
                u - model.m * x_dotdot_true - model.c * x_dot_true - model.k * x_true
            ) / model.alpha
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
            z_param_loss, z_phy_loss, z_dot_loss = (
                z_learned[sta_ind_loss:end_ind_loss],
                z_phy[sta_ind_loss:end_ind_loss],
                z_dot_learned[sta_ind_loss:end_ind_loss],
            )

        print(f"Total time: {time.time()-time0: .4f} s")

        plt.figure()
        plt.plot(loss_history)
        plt.title(r"Training Loss")
        plt.xlabel(r"Epoch")
        plt.ylabel(r"Loss Value")
        plt.tight_layout()
        plt.savefig(path + f"Training_Loss_Final.png")
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
        plt.title(r"$x(t)$ Comparison")
        plt.xlabel(r"Time step")
        plt.ylabel(r"$x$ value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path + f"x_t_Comparison_Epoch_Final.png")
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
        plt.title(r"$\dot{x}(t)$ Comparison")
        plt.xlabel(r"Time step")
        plt.ylabel(r"$\dot{x}$ value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path + f"x_dot_t_Comparison_Epoch_Final.png")
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
        plt.title(r"Learned $z(t)$ Trajectory")
        plt.xlabel(r"Time step")
        plt.ylabel(r"$z$ value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path + f"Learned_z_t_Trajectory_Epoch_Final.png")
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
        plt.title(r"Learned $\dot{z}(t)$ Trajectory")
        plt.xlabel(r"Time step")
        plt.ylabel(r"$\dot{z}$ value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path + f"Learned_z_dot_t_Trajectory_Epoch_Final.png")
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
            z_true=z_true.cpu().numpy(),
            z_learned=z_learned.detach().cpu().numpy(),
            z_phy=z_phy.detach().cpu().numpy(),
            z_dot_true=z_dot_true.cpu().numpy(),
            z_dot_learned=z_dot_learned.detach().cpu().numpy(),
            m=float(model.m),
            c=float(model.c),
            k=float(model.k),
            alpha=float(model.alpha),
        )

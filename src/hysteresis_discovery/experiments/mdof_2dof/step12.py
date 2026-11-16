"""STEP12: mdof 2dof."""

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

    from ...io import load_npz

    with experiment_session(config, "mdof_2dof", "step12") as session:
        path = str(session.output_dir) + "/"
        device = session.device
        time0 = time.time()

        data = load_npz(session.data_dir / "mdof_2dof_truth_data.npz")

        M_true = data["M"]
        m1_true = M_true[0][0]
        m2_true = M_true[1][1]
        C_true = data["C"]
        c11_true = C_true[0][0]
        c12_true = C_true[0][1]
        c21_true = C_true[1][0]
        c22_true = C_true[1][1]
        K_true = data["K"]
        k11_true = K_true[0][0]
        k12_true = K_true[0][1]
        k21_true = K_true[1][0]
        k22_true = K_true[1][1]
        H_true = data["H"]
        h1_true = H_true[0][0]
        h2_true = H_true[1][0]
        B_true = data["B"]
        a_true = float(data["a"])
        b_true = float(data["b"])
        U0_true = float(data["U0"])
        f0_true = float(data["f0"])
        t0_true = float(data["t0"])
        tf_true = float(data["tf"])
        dt_true = float(data["dt"])
        y0_true = data["y0"]

        t_true = data["t"]
        u_true = data["u"]
        x1_true = data["x1"]
        x2_true = data["x2"]
        v1_true = data["v1"]
        v2_true = data["v2"]
        a1_true = data["a1"]
        a2_true = data["a2"]
        z_true = data["z"]
        zdot_true = data["zdot"]

        t_train_true = data["t_train"]
        x1_train_true = data["x1_train"]
        x2_train_true = data["x2_train"]
        v1_train_true = data["v1_train"]
        v2_train_true = data["v2_train"]
        a1_train_true = data["a1_train"]
        a2_train_true = data["a2_train"]
        z_train_true = data["z_train"]
        zdot_train_true = data["zdot_train"]
        u_train_true = data["u_train"]

        t_test_true = data["t_test"]
        x1_test_true = data["x1_test"]
        x2_test_true = data["x2_test"]
        v1_test_true = data["v1_test"]
        v2_test_true = data["v2_test"]
        a1_test_true = data["a1_test"]
        a2_test_true = data["a2_test"]
        z_test_true = data["z_test"]
        zdot_test_true = data["zdot_test"]
        u_test_true = data["u_test"]

        # Step 1: Assume z = 0 and learn [m, c, k] under the linearized model: m*x'' + c*x' + k*x = u
        # a1 = (u - c11*v1 - c12*v2 - k11*x1 - k12*x2 - h1*z) / m1;  m1*a1 + c11*v1 + c12*v2 + k11*x1 + k12*x2 + h1*z = u
        # a2 = (u - c21*v1 - c22*v2 - k21*x1 - k22*x2 - h2*z) / m2;  m2*a2 + c21*v1 + c22*v2 + k21*x1 + k22*x2 + h2*z = u

        m1_0 = m1_true * 0.95
        m2_0 = m2_true * 1.05
        c11_0 = c11_true * 1.05
        c12_0 = c12_true * 0.95
        c21_0 = c21_true * 1.05
        c22_0 = c22_true * 0.95
        k11_0 = k11_true * 0.95
        k12_0 = k12_true * 1.05
        k21_0 = k21_true * 0.95
        k22_0 = k22_true * 1.05
        h1_0 = h1_true * 0.95
        h2_0 = h2_true * 1.05

        # Step 2: Learn z
        print(
            "###################################### Step 2 ######################################"
        )

        t_train_true_tensor = torch.from_numpy(t_train_true).to(device=device).float()
        u_train_true_tensor = torch.from_numpy(u_train_true).to(device=device).float()
        x1_train_true_tensor = torch.from_numpy(x1_train_true).to(device=device).float()
        x2_train_true_tensor = torch.from_numpy(x2_train_true).to(device=device).float()
        v1_train_true_tensor = torch.from_numpy(v1_train_true).to(device=device).float()
        v2_train_true_tensor = torch.from_numpy(v2_train_true).to(device=device).float()
        a1_train_true_tensor = torch.from_numpy(a1_train_true).to(device=device).float()
        a2_train_true_tensor = torch.from_numpy(a2_train_true).to(device=device).float()
        z_train_true_tensor = torch.from_numpy(z_train_true).to(device=device).float()
        zdot_train_true_tensor = (
            torch.from_numpy(zdot_train_true).to(device=device).float()
        )
        t0 = t_train_true_tensor[0]
        z0 = (
            u_train_true_tensor
            - float(m1_0) * a1_train_true_tensor
            - float(c11_0) * v1_train_true_tensor
            - float(c12_0) * v2_train_true_tensor
            - float(k11_0) * x1_train_true_tensor
            - float(k12_0) * x2_train_true_tensor
        ) / float(h1_0)

        class LearnableZMDOF(nn.Module):
            def __init__(self, hidden_dim=128):
                super().__init__()
                self.m1 = nn.Parameter(
                    torch.tensor(float(m1_0), device=device, dtype=torch.float32)
                )
                self.m2 = nn.Parameter(
                    torch.tensor(float(m2_0), device=device, dtype=torch.float32)
                )
                self.c11 = nn.Parameter(
                    torch.tensor(float(c11_0), device=device, dtype=torch.float32)
                )
                self.c12 = nn.Parameter(
                    torch.tensor(float(c12_0), device=device, dtype=torch.float32)
                )
                self.c21 = nn.Parameter(
                    torch.tensor(float(c21_0), device=device, dtype=torch.float32)
                )
                self.c22 = nn.Parameter(
                    torch.tensor(float(c22_0), device=device, dtype=torch.float32)
                )
                self.k11 = nn.Parameter(
                    torch.tensor(float(k11_0), device=device, dtype=torch.float32)
                )
                self.k12 = nn.Parameter(
                    torch.tensor(float(k12_0), device=device, dtype=torch.float32)
                )
                self.k21 = nn.Parameter(
                    torch.tensor(float(k21_0), device=device, dtype=torch.float32)
                )
                self.k22 = nn.Parameter(
                    torch.tensor(float(k22_0), device=device, dtype=torch.float32)
                )
                self.h1 = nn.Parameter(
                    torch.tensor(float(h1_0), device=device, dtype=torch.float32)
                )
                self.h2 = nn.Parameter(
                    torch.tensor(float(h2_0), device=device, dtype=torch.float32)
                )

                self.z_param = nn.Parameter(z0)
                self.z_grad = None

            def update_z_grad(self, dt):
                self.z_grad = diff_series(self.z_param, dt)

            def forward(self, t, state):
                x1, x2, v1, v2, z = state
                f = lerp_series(u_train_true_tensor, t, t0, dt_true)
                z = lerp_series(self.z_param, t, t0, dt_true)
                dx1 = v1
                dx2 = v2
                dv1 = (
                    f
                    - self.c11 * v1
                    - self.c12 * v2
                    - self.k11 * x1
                    - self.k12 * x2
                    - self.h1 * z
                ) / self.m1
                dv2 = (
                    f
                    - self.c21 * v1
                    - self.c22 * v2
                    - self.k21 * x1
                    - self.k22 * x2
                    - self.h2 * z
                ) / self.m2
                assert (
                    self.z_grad is not None
                ), "call model.update_z_grad(dt) before odeint"
                dz = lerp_series(self.z_grad, t, t0, dt_true)
                return torch.stack([dx1, dx2, dv1, dv2, dz])

        model = LearnableZMDOF().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3e-4, betas=(0.9, 0.98), weight_decay=1e-4
        )
        epochs = config.epochs
        loss_history = []

        ep_str = f" (Epoch {epochs})"
        for epoch in range(epochs):
            model.update_z_grad(dt_true)
            x0 = torch.stack(
                [
                    x1_train_true_tensor[0],
                    x2_train_true_tensor[0],
                    v1_train_true_tensor[0],
                    v2_train_true_tensor[0],
                    model.z_param[0].detach(),
                ]
            ).to(device)
            pred = odeint(model, x0, t_train_true_tensor, method="rk4")
            x1_pred, x2_pred, v1_pred, v2_pred, z_pred = (
                pred[:, 0],
                pred[:, 1],
                pred[:, 2],
                pred[:, 3],
                pred[:, 4],
            )
            z_learned = model.z_param
            z_dot_learned = diff_series(z_learned, dt_true)

            RMSE_x1 = rmse(x1_pred, x1_train_true_tensor)
            NRMSE_x1 = RMSE_x1 / (
                x1_train_true_tensor.max() - x1_train_true_tensor.min() + 1e-8
            )
            RMSE_x2 = rmse(x2_pred, x2_train_true_tensor)
            NRMSE_x2 = RMSE_x2 / (
                x2_train_true_tensor.max() - x2_train_true_tensor.min() + 1e-8
            )
            RMSE_v1 = rmse(v1_pred, v1_train_true_tensor)
            NRMSE_v1 = RMSE_v1 / (
                v1_train_true_tensor.max() - v1_train_true_tensor.min() + 1e-8
            )
            RMSE_v2 = rmse(v2_pred, v2_train_true_tensor)
            NRMSE_v2 = RMSE_v2 / (
                v2_train_true_tensor.max() - v2_train_true_tensor.min() + 1e-8
            )

            RMSE_z_param = rmse(z_learned, z_pred)
            NRMSE_z_param = RMSE_z_param / (z_pred.max() - z_pred.min() + 1e-8)

            lam_z = 1e-3
            z_second = z_learned[2:] - 2 * z_learned[1:-1] + z_learned[:-2]
            smooth_penalty = (z_second**2).mean()

            loss = NRMSE_x1 + NRMSE_x2 + NRMSE_v1 + NRMSE_v2

            RMSE_z = rmse(z_learned, z_train_true_tensor)
            NRMSE_z = RMSE_z / (
                z_train_true_tensor.max() - z_train_true_tensor.min() + 1e-8
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_history.append(loss.item())
            if (epoch == 0) or ((epoch + 1) % 5 == 0):
                print(
                    f"Epoch {epoch + 1} | NRMSE Loss: {loss.item():.4e} | "
                    f"NRMSE_x1: {NRMSE_x1.item():.4f} | NRMSE_x2: {NRMSE_x2.item():.4f} | "
                    f"NRMSE_v1: {NRMSE_v1.item():.4f} | NRMSE_v2: {NRMSE_v2.item():.4f} | "
                    f"NRMSE_z_param: {NRMSE_z_param.item():.4f} | NRMSE_z: {NRMSE_z.item():.4f} | "
                    f"m1: {model.m1.item():.4f} (true: {m1_true:.1f}) | m2: {model.m2.item():.4f} (true: {m2_true:.1f}) | "
                    f"c11: {model.c11.item():.4f} (true: {c11_true:.1f}) | c12: {model.c12.item():.4f} (true: {c12_true:.1f}) | "
                    f"c21: {model.c21.item():.4f} (true: {c21_true:.1f}) | c22: {model.c22.item():.4f} (true: {c22_true:.1f}) | "
                    f"k11: {model.k11.item():.4f} (true: {k11_true:.1f}) | k12: {model.k12.item():.4f} (true: {k12_true:.1f}) | "
                    f"k21: {model.k21.item():.4f} (true: {k21_true:.1f}) | k22: {model.k22.item():.4f} (true: {k22_true:.1f}) | "
                    f"h1: {model.h1.item():.4f} (true: {h1_true:.1f}) | h2: {model.h2.item():.4f} (true: {h2_true:.1f})"
                )

            if session.checkpoint(
                epoch, 5, include_first=True, interval=config.plot_interval
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
                    t_train_true_tensor.detach().cpu(),
                    x1_train_true_tensor.detach().cpu(),
                    label=r"True $x_1(t)$",
                )
                plt.plot(
                    t_train_true_tensor.detach().cpu(),
                    x1_pred.detach().cpu(),
                    "--",
                    label=r"Predicted $x_1(t)$",
                )
                plt.title(r"$x_1(t)$ Comparison" + ep_str)
                plt.xlabel(r"Time step")
                plt.ylabel(r"$x_1$ value")
                plt.legend()
                plt.tight_layout()
                plt.savefig(path + f"x1_t_Comparison_Epoch_{epoch + 1}.png", dpi=600)
                plt.close()

                plt.figure()
                plt.plot(
                    t_train_true_tensor.detach().cpu(),
                    x2_train_true_tensor.detach().cpu(),
                    label=r"True $x_2(t)$",
                )
                plt.plot(
                    t_train_true_tensor.detach().cpu(),
                    x2_pred.detach().cpu(),
                    "--",
                    label=r"Predicted $x_2(t)$",
                )
                plt.title(r"$x_2(t)$ Comparison" + ep_str)
                plt.xlabel(r"Time step")
                plt.ylabel(r"$x_2$ value")
                plt.legend()
                plt.tight_layout()
                plt.savefig(path + f"x2_t_Comparison_Epoch_{epoch + 1}.png", dpi=600)
                plt.close()

                plt.figure()
                plt.plot(
                    t_train_true_tensor.detach().cpu(),
                    v1_train_true_tensor.detach().cpu(),
                    label=r"True $v_1(t)$",
                )
                plt.plot(
                    t_train_true_tensor.detach().cpu(),
                    v1_pred.detach().cpu(),
                    "--",
                    label=r"Predicted $v_1(t)$",
                )
                plt.title(r"$v_1(t)$ Comparison" + ep_str)
                plt.xlabel(r"Time step")
                plt.ylabel(r"$v_1$ value")
                plt.legend()
                plt.tight_layout()
                plt.savefig(path + f"v1_t_Comparison_Epoch_{epoch + 1}.png", dpi=600)
                plt.close()

                plt.figure()
                plt.plot(
                    t_train_true_tensor.detach().cpu(),
                    v2_train_true_tensor.detach().cpu(),
                    label=r"True $v_2(t)$",
                )
                plt.plot(
                    t_train_true_tensor.detach().cpu(),
                    v2_pred.detach().cpu(),
                    "--",
                    label=r"Predicted $v_2(t)$",
                )
                plt.title(r"$v_2(t)$ Comparison" + ep_str)
                plt.xlabel(r"Time step")
                plt.ylabel(r"$v_2$ value")
                plt.legend()
                plt.tight_layout()
                plt.savefig(path + f"v2_t_Comparison_Epoch_{epoch + 1}.png", dpi=600)
                plt.close()

                plt.figure()
                plt.plot(
                    t_train_true_tensor.detach().cpu(),
                    z_train_true_tensor.detach().cpu(),
                    label=r"True $z(t)$",
                )
                plt.plot(
                    t_train_true_tensor.detach().cpu(),
                    z_learned.detach().cpu(),
                    "--",
                    label=r"Predicted $z(t)$",
                )
                plt.title(r"$z(t)$ Comparison" + ep_str)
                plt.xlabel(r"Time step")
                plt.ylabel(r"$z$ value")
                plt.legend()
                plt.tight_layout()
                plt.savefig(path + f"z_t_Comparison_Epoch_{epoch + 1}.png", dpi=600)
                plt.close()

                np.savez(
                    path + f"data_Epoch_{epoch + 1}.npz",
                    t=t_train_true_tensor.cpu().numpy(),
                    u=u_train_true_tensor.cpu().numpy(),
                    x0=x0.cpu().numpy(),
                    x1_true=x1_train_true_tensor.cpu().numpy(),
                    x1_pred=x1_pred.detach().cpu().numpy(),
                    x2_true=x2_train_true_tensor.cpu().numpy(),
                    x2_pred=x2_pred.detach().cpu().numpy(),
                    v1_true=v1_train_true_tensor.cpu().numpy(),
                    v1_pred=v1_pred.detach().cpu().numpy(),
                    v2_true=v2_train_true_tensor.cpu().numpy(),
                    v2_pred=v2_pred.detach().cpu().numpy(),
                    z_true=z_train_true_tensor.cpu().numpy(),
                    z_pred=z_learned.detach().cpu().numpy(),
                    z_dot_true=zdot_train_true_tensor.cpu().numpy(),
                    z_dot_pred=z_dot_learned.detach().cpu().numpy(),
                    m1=float(model.m1),
                    m2=float(model.m2),
                    c11=float(model.c11),
                    c12=float(model.c12),
                    c21=float(model.c21),
                    c22=float(model.c22),
                    k11=float(model.k11),
                    k12=float(model.k12),
                    k21=float(model.k21),
                    k22=float(model.k22),
                    h1=float(model.h1),
                    h2=float(model.h2),
                )

        with torch.no_grad():
            model.update_z_grad(dt_true)
            x0 = torch.tensor(
                [
                    x1_train_true_tensor[0],
                    x2_train_true_tensor[0],
                    v1_train_true_tensor[0],
                    v2_train_true_tensor[0],
                    model.z_param[0].detach(),
                ],
                device=device,
            )
            pred = odeint(model, x0, t_train_true_tensor, method="rk4")
            x1_pred, x2_pred, v1_pred, v2_pred, z_pred = (
                pred[:, 0],
                pred[:, 1],
                pred[:, 2],
                pred[:, 3],
                pred[:, 4],
            )
            z_learned = model.z_param
            z_dot_learned = diff_series(z_learned, dt_true)

        print(f"Total time: {time.time() - time0: .4f} s")

        np.savez(
            path + f"data_Final.npz",
            t=t_train_true_tensor.cpu().numpy(),
            u=u_train_true_tensor.cpu().numpy(),
            x0=x0.cpu().numpy(),
            x1_true=x1_train_true_tensor.cpu().numpy(),
            x1_pred=x1_pred.detach().cpu().numpy(),
            x2_true=x2_train_true_tensor.cpu().numpy(),
            x2_pred=x2_pred.detach().cpu().numpy(),
            v1_true=v1_train_true_tensor.cpu().numpy(),
            v1_pred=v1_pred.detach().cpu().numpy(),
            v2_true=v2_train_true_tensor.cpu().numpy(),
            v2_pred=v2_pred.detach().cpu().numpy(),
            z_true=z_train_true_tensor.cpu().numpy(),
            z_pred=z_learned.detach().cpu().numpy(),
            z_dot_true=zdot_train_true_tensor.cpu().numpy(),
            z_dot_pred=z_dot_learned.detach().cpu().numpy(),
            m1=float(model.m1),
            m2=float(model.m2),
            c11=float(model.c11),
            c12=float(model.c12),
            c21=float(model.c21),
            c22=float(model.c22),
            k11=float(model.k11),
            k12=float(model.k12),
            k21=float(model.k21),
            k22=float(model.k22),
            h1=float(model.h1),
            h2=float(model.h2),
        )

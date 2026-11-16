"""STEP3: nonlinear fractional."""

from ...runtime import RunConfig, experiment_session


def run(config: RunConfig) -> None:
    """Run the experiment with shared paths and runtime settings."""
    import numpy as np
    import matplotlib.pyplot as plt
    import sympy
    from sklearn.metrics import mean_squared_error
    import time
    from ...symbolic import evaluate_expression, select_expression

    from ...io import load_npz

    with experiment_session(config, "nonlinear_fractional", "step3") as session:
        path = str(session.output_dir) + "/"
        time0 = time.time()

        fs = 100
        m_true = 1.0
        c_true = 0.8
        k_true = 0.5
        alpha_true = 1.0
        A_true = 4.0
        beta_true = 5.0
        gamma_true = -4.0
        n_true = 1.5

        # Step 3: Substitute the learned z and z' into SR to obtain explicit expressions
        print(
            "###################################### Step 3 ######################################"
        )
        npzfile = load_npz(session.output_dir / "data_Final.npz")

        from pysr import PySRRegressor

        t_file = npzfile["t"]
        u_file = npzfile["u"]
        x0_file = npzfile["x0"]
        x_true_file = npzfile["x_true"]
        x_pred_file = npzfile["x_pred"]
        x_dot_true_file = npzfile["x_dot_true"]
        x_dot_pred_file = npzfile["x_dot_pred"]
        x_dotdot_true_file = npzfile["x_dotdot_true"]
        x_dotdot_pred_file = npzfile["x_dotdot_pred"]
        z_true_file = npzfile["z_true"]
        z_learned_file = npzfile["z_learned"]
        z_dot_true_file = npzfile["z_dot_true"]
        z_dot_learned_file = npzfile["z_dot_learned"]
        m_file = npzfile["m"]
        c_file = npzfile["c"]
        k_file = npzfile["k"]
        alpha_file = npzfile["alpha"]

        start_index = 2 * fs
        end_index = 6 * fs
        data_t_SR = t_file[start_index:end_index]
        data_u_SR = u_file[start_index:end_index]
        data_x_SR = x_pred_file[start_index:end_index]
        data_x_true = x_true_file[start_index:end_index]
        data_x_dot_SR = x_dot_pred_file[start_index:end_index]
        data_x_dot_true = x_dot_true_file[start_index:end_index]
        data_x_dotdot_SR = x_dotdot_pred_file[start_index:end_index]
        data_x_dotdot_true = x_dotdot_true_file[start_index:end_index]
        data_z_SR = z_learned_file[start_index:end_index]
        data_z_true = z_true_file[start_index:end_index]
        data_z_dot_SR = z_dot_learned_file[start_index:end_index]
        data_z_dot_true = z_dot_true_file[start_index:end_index]
        print("MSE(x) =", mean_squared_error(x_pred_file, x_true_file))
        print("MSE(x_dot) =", mean_squared_error(x_dot_pred_file, x_dot_true_file))
        print(
            "MSE(x_dotdot) =",
            mean_squared_error(x_dotdot_pred_file, x_dotdot_true_file),
        )
        print("MSE(z) =", mean_squared_error(z_learned_file, z_true_file))
        print("MSE(z_dot) =", mean_squared_error(z_dot_learned_file, z_dot_true_file))

        plt.figure()
        plt.plot(data_t_SR, data_u_SR, color="gray", linewidth=2)
        plt.xlabel(r"Time step $t$")
        plt.ylabel(r"External force $u$")
        plt.title(r"Solution of Nonlinear Bouc-Wen System")
        plt.grid(True)
        plt.tight_layout()

        plt.figure()
        plt.plot(data_t_SR, data_x_true, color="gray", linewidth=2, label=r"Truth")
        plt.plot(data_t_SR, data_x_SR, "--", label=r"Prediction")
        plt.xlabel(r"Time step $t$")
        plt.ylabel(r"Displacement $x$")
        plt.title(
            r"Solution of Nonlinear Bouc-Wen System, MSE=%.4f"
            % mean_squared_error(data_x_true, data_x_SR)
        )
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.figure()
        plt.plot(data_t_SR, data_x_dot_true, color="gray", linewidth=2, label=r"Truth")
        plt.plot(data_t_SR, data_x_dot_SR, "--", label=r"Prediction")
        plt.xlabel(r"Time step $t$")
        plt.ylabel(r"Velocity $\dot{x}$")
        plt.title(
            r"Solution of Nonlinear Bouc-Wen System, MSE=%.4f"
            % mean_squared_error(data_x_dot_true, data_x_dot_SR)
        )
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.figure()
        plt.plot(data_t_SR, data_z_true, color="gray", linewidth=2, label=r"Truth")
        plt.plot(data_t_SR, data_z_SR, "--", label=r"Prediction")
        plt.xlabel(r"Time step $t$")
        plt.ylabel(r"Hysteretic force $z$")
        plt.title(
            r"Solution of Nonlinear Bouc-Wen System, MSE=%.4f"
            % mean_squared_error(data_z_true, data_z_SR)
        )
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.figure()
        plt.plot(data_t_SR, data_z_dot_true, color="gray", linewidth=2, label=r"Truth")
        plt.plot(data_t_SR, data_z_dot_SR, "--", label=r"Prediction")
        plt.xlabel(r"Time step $t$")
        plt.ylabel(r"Hysteretic force derivative $\dot{z}$")
        plt.title(
            r"Solution of Nonlinear Bouc-Wen System, MSE=%.4f"
            % mean_squared_error(data_z_dot_true, data_z_dot_SR)
        )
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.figure()
        plt.plot(
            data_x_true, data_x_dot_true, color="gray", linewidth=2, label=r"Truth"
        )
        plt.plot(data_x_SR, data_x_dot_SR, "--", label=r"Prediction")
        plt.xlabel(r"Displacement $x$")
        plt.ylabel(r"Hysteretic force $\dot{x}$")
        plt.title(r"$x$-$\dot{x}$ Hysteresis loop learned from solver-based model")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.figure()
        plt.plot(data_x_true, data_z_true, color="gray", linewidth=2, label=r"Truth")
        plt.plot(data_x_SR, data_z_SR, "--", label=r"Prediction")
        plt.xlabel(r"Displacement $x$")
        plt.ylabel(r"Hysteretic force $z$")
        plt.title(r"$x$-$z$ Hysteresis loop learned from solver-based model")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        ###############################  The hysteresis link can be run multiple times to learn the residual again #######################
        z_dot_expr = sympy.S(0)
        z_dot_pred = np.zeros_like(data_z_dot_SR)

        for i in range(1):  # Number of repetitions
            model = PySRRegressor(
                **session.symbolic_options(config),
                niterations=config.niterations,
                early_stop_condition=1e-6,
                unary_operators=["abs"],
                binary_operators=["+", "-", "*", "abs_pow(x,n)=abs(x)^n"],
                extra_sympy_mappings={
                    "abs_pow": lambda x, n: sympy.Abs(x) ** n,
                },
                constraints={
                    "abs_pow": (2, 1),
                },
                nested_constraints={
                    "abs_pow": {"abs_pow": 0},
                },
                complexity_of_constants=1,
                complexity_of_variables=2,
                complexity_of_operators={
                    "+": 1,
                    "-": 1,
                    "*": 1,
                    "abs": 1,
                    "abs_pow": 1,
                },
                verbosity=1,
            )

            model.fit(
                np.stack((data_x_dot_SR, data_z_SR), axis=1),
                data_z_dot_SR - z_dot_pred,
                variable_names=["x_dot", "z"],
            )

            variables = ["x_dot", "z"]
            best_expr, best_score = select_expression(model, variables)
            print(f"Best expression after filtering: {best_expr}, score={best_score}")

            z_dot_expr += best_expr
            z_dot_expr = sympy.simplify(
                sympy.powsimp(sympy.together(sympy.expand(z_dot_expr)), force=True)
            )

            z_dot_pred = evaluate_expression(
                z_dot_expr, variables, data_x_dot_SR, data_z_SR
            )

            mse_z_dot = mean_squared_error(z_dot_pred, data_z_dot_SR)
            print(f"PySR Mean Squared Error for z_dot: {mse_z_dot:.10f}")

        print("Step 3: Learned expression z_dot(x_dot, z) =", z_dot_expr)

        n_1 = n_true - 1
        true_expr = f"{A_true}*x_dot - {beta_true}*|x_dot|*|z|**{n_1}*z - {gamma_true}*x_dot*|z|**{n_true}"
        print("True expression z_dot(x_dot, z) =", true_expr)
        true_expr = f"{A_true}*x_dot - |z|**{n_1} * ({beta_true}*|x_dot|*z + {gamma_true}*x_dot*|z|)"
        print("True expression z_dot(x_dot, z) =", true_expr)

        plt.figure()
        plt.plot(
            data_t_SR,
            data_z_dot_SR,
            label=r"Learned from Step 2",
            color="gray",
            linewidth=2,
        )
        plt.plot(
            data_t_SR,
            z_dot_pred,
            label=r"SR predicted results",
            color="green",
            linestyle="--",
        )
        plt.xlabel(r"Time step")
        plt.ylabel(r"$\dot{z}$ value")
        plt.title(r"Comparison of true and predicted solutions for $\dot{z}$")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path + f"SR result_z_dot.png", dpi=600)
        plt.show()

        ###############################  The motion equation can be run multiple times to learn the residual again #######################
        x_dotdot_expr = sympy.S(0)
        x_dotdot_pred = np.zeros_like(data_x_SR)
        for i in range(1):
            model = PySRRegressor(
                **session.symbolic_options(config),
                niterations=config.niterations,
                early_stop_condition=1e-6,
                binary_operators=["+", "-", "*", "pow"],
                extra_sympy_mappings={"pow": sympy.Pow},
                complexity_of_constants=1,
                complexity_of_variables=2,
                constraints={"pow": (2, 1)},
                verbosity=1,
            )

            model.fit(
                np.stack((data_x_SR, data_x_dot_SR, data_z_SR, data_u_SR), axis=1),
                data_x_dotdot_SR - x_dotdot_pred,
                variable_names=["x", "x_dot", "z", "u"],
            )

            variables = ["x", "x_dot", "z", "u"]
            best_expr, best_score = select_expression(model, variables)
            print(f"Best expression after filtering: {best_expr}, score={best_score}")

            x_dotdot_expr += best_expr
            x_dotdot_expr = sympy.simplify(
                sympy.powsimp(sympy.together(sympy.expand(x_dotdot_expr)), force=True)
            )

            x_dotdot_pred = evaluate_expression(
                x_dotdot_expr, variables, data_x_SR, data_x_dot_SR, data_z_SR, data_u_SR
            )

            mse_x_dotdot = mean_squared_error(x_dotdot_pred, data_x_dotdot_SR)
            print(f"PySR Mean Squared Error for x_dotdot: {mse_x_dotdot:.10f}")

        print("Step 3: Learned expression x_dotdot(x, x_dot, z, u) =", x_dotdot_expr)
        print("Step 3: Learned expression z_dot(x_dot, z) =", z_dot_expr)

        true_expr = f"u - {c_true/m_true} * x_dot - {k_true/m_true} * x**3 - {alpha_true/m_true} * z"
        print("True expression x_dotdot(x, x_dot, z, u) =", true_expr)
        true_expr = f"{A_true}*x_dot - {beta_true}*|x_dot|*|z|**{n_1}*z - {gamma_true}*x_dot*|z|**{n_true}"
        print("True expression z_dot(x_dot, z) =", true_expr)
        true_expr = f"{A_true}*x_dot - |z|**{n_1} * ({beta_true}*|x_dot|*z + {gamma_true}*x_dot*|z|)"
        print("True expression z_dot(x_dot, z) =", true_expr)

        print(
            "m_file, c_file, k_file, alpha_file =", m_file, c_file, k_file, alpha_file
        )

        plt.figure()
        plt.plot(
            data_t_SR,
            data_x_dotdot_SR,
            label=r"Learned from Step 2",
            color="gray",
            linewidth=2,
        )
        plt.plot(
            data_t_SR,
            x_dotdot_pred,
            label=r"SR predicted results",
            color="green",
            linestyle="--",
        )
        plt.xlabel(r"Time step")
        plt.ylabel(r"$\ddot{x}$ value")
        plt.title(r"Comparison of true and predicted solutions for $\ddot{x}$")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path + f"SR result_x_dotdot.png", dpi=600)
        plt.show()

        session.save_equations(z_dot=z_dot_expr, x_dotdot=x_dotdot_expr)
        np.savez(
            session.output_dir / "data_step3.npz",
            data_t_SR=data_t_SR,
            data_x_SR=data_x_SR,
            data_x_dot_SR=data_x_dot_SR,
            data_u_SR=data_u_SR,
            data_z_SR=data_z_SR,
            data_z_dot_SR=data_z_dot_SR,
            z_dot_pred=z_dot_pred,
            data_x_dotdot_SR=data_x_dotdot_SR,
            x_dotdot_pred=x_dotdot_pred,
        )

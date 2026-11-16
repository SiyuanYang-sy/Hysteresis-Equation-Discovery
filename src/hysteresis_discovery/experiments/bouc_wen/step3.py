"""STEP3: bouc wen."""

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

    with experiment_session(config, "bouc_wen", "step3") as session:
        path = str(session.output_dir) + "/"
        time0 = time.time()

        # Step 3: Substitute the learned z and z' into SR to obtain an expression
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
        z_true_file = npzfile["z_true"]
        z_learned_file = npzfile["z_learned"]
        z_phy_file = npzfile["z_phy"]
        z_dot_true_file = npzfile["z_dot_true"]
        z_dot_learned_file = npzfile["z_dot_learned"]
        m_file = npzfile["m"]
        c_file = npzfile["c"]
        k_file = npzfile["k"]
        alpha_file = npzfile["alpha"]

        start_index = 10
        end_index = -5
        data_x_dotdot_LSM = x_dotdot_true_file[start_index:end_index]
        data_x_dot_LSM = x_dot_pred_file[start_index:end_index]
        data_x_LSM = x_pred_file[start_index:end_index]
        data_z_LSM = z_learned_file[start_index:end_index]
        data_u_LSM = u_file[start_index:end_index]

        X_matrix = np.vstack(
            (data_x_dotdot_LSM, data_x_dot_LSM, data_x_LSM, data_z_LSM)
        ).T
        Y_vector = data_u_LSM
        para_mckalpha = np.linalg.lstsq(X_matrix, Y_vector, rcond=None)[0]
        m_para_LSM, c_para_LSM, k_para_LSM, alpha_para_LSM = para_mckalpha
        print(
            f"Learned Parameters [m, c, k, alpha]: "
            f"m_para={m_para_LSM:.4f}, c_para={c_para_LSM:.4f}, k_para={k_para_LSM:.4f}, alpha_para={alpha_para_LSM:.4f}"
        )

        start_index = 10
        end_index = -5
        data_t_SR = t_file[start_index:end_index]
        data_x_SR = x_pred_file[start_index:end_index]
        data_x_dot_SR = x_dot_pred_file[start_index:end_index]
        data_z_SR = z_learned_file[start_index:end_index]
        data_z_dot_SR = z_dot_learned_file[start_index:end_index]

        model = PySRRegressor(
            **session.symbolic_options(config),
            niterations=config.niterations,  # Increase the number of iterations to improve the fitting performance
            unary_operators=["abs"],
            binary_operators=[
                "+",
                "-",
                "*",
                "abs_pow(x, n) = abs(x)^n",
            ],  # Avoid division
            extra_sympy_mappings={"abs_pow": lambda x, n: sympy.Abs(x) ** n},
            constraints={
                "abs": 1,
                "abs_pow": (1, 1),
            },  # Restrict n in abs_pow(x, n) to be a constant
            nested_constraints={
                "abs": {"abs": 0, "abs_pow": 0},  # Disallow nesting
                "abs_pow": {"abs": 0, "abs_pow": 0},  # Disallow nesting
            },
            verbosity=1,  # Minimum console output: 0, 1, or 2
        )

        model.fit(
            np.stack((data_x_dot_SR, data_z_SR), axis=1),
            data_z_dot_SR,
            variable_names=["x_dot", "z"],
        )

        print(time.time() - time0, "s")

        variables = ["x_dot", "z"]
        # Sort by score in descending order (a higher score is generally better)
        best_expr, best_score = select_expression(model, variables)
        print(f"Best expression after filtering: {best_expr}, score={best_score}")

        z_dot_expr = best_expr

        z_dot_pred = evaluate_expression(
            z_dot_expr, variables, data_x_dot_SR, data_z_SR
        )

        mse_z_dot = mean_squared_error(z_dot_pred, data_z_dot_SR)
        print(f"PySR Mean Squared Error for z_dot: {mse_z_dot:.10f}")

        print("Step 3: Learned expression z_dot(x_dot, z) =", z_dot_expr)
        print("True expression: 50000*x_dot - 800*Abs(x_dot)*z + 1100*x_dot*Abs(z)")

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
        plt.savefig(path + f"SR result.png", dpi=600)
        plt.show()

        np.savez(
            path + f"data_step3.npz",
            data_x_dot_SR=data_x_dot_SR,
            data_z_SR=data_z_SR,
            data_t_SR=data_t_SR,
            data_z_dot_SR=data_z_dot_SR,
            z_dot_pred=z_dot_pred,
        )

        session.save_equations(z_dot=z_dot_expr)

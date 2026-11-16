"""STEP3: mdof 2dof."""

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

    with experiment_session(config, "mdof_2dof", "step3") as session:
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
        x1_true_file = npzfile["x1_true"]
        x1_pred_file = npzfile["x1_pred"]
        x2_true_file = npzfile["x2_true"]
        x2_pred_file = npzfile["x2_pred"]
        v1_true_file = npzfile["v1_true"]
        v1_pred_file = npzfile["v1_pred"]
        v2_true_file = npzfile["v2_true"]
        v2_pred_file = npzfile["v2_pred"]
        z_true_file = npzfile["z_true"]
        z_pred_file = npzfile["z_pred"]
        z_dot_true_file = npzfile["z_dot_true"]
        z_dot_pred_file = npzfile["z_dot_pred"]
        m1_file = float(npzfile["m1"])
        m2_file = float(npzfile["m2"])
        c11_file = float(npzfile["c11"])
        c12_file = float(npzfile["c12"])
        c21_file = float(npzfile["c21"])
        c22_file = float(npzfile["c22"])
        k11_file = float(npzfile["k11"])
        k12_file = float(npzfile["k12"])
        k21_file = float(npzfile["k21"])
        k22_file = float(npzfile["k22"])
        h1_file = float(npzfile["h1"])
        h2_file = float(npzfile["h2"])

        start_index = 0
        end_index = -1
        data_t_SR = t_file[start_index:end_index]
        data_x1_SR = x1_pred_file[start_index:end_index]
        data_x2_SR = x2_pred_file[start_index:end_index]
        data_v1_SR = v1_pred_file[start_index:end_index]
        data_v2_SR = v2_pred_file[start_index:end_index]
        data_z_SR = z_pred_file[start_index:end_index]
        data_z_dot_SR = z_dot_pred_file[start_index:end_index]

        model = PySRRegressor(
            **session.symbolic_options(config),
            niterations=config.niterations,
            unary_operators=["abs"],
            binary_operators=["+", "-", "*", "abs_pow(x, n) = abs(x)^n"],
            extra_sympy_mappings={"abs_pow": lambda x, n: sympy.Abs(x) ** n},
            constraints={"abs": 1, "abs_pow": (1, 1)},
            nested_constraints={
                "abs": {"abs": 0, "abs_pow": 0},
                "abs_pow": {"abs": 0, "abs_pow": 0},
            },
            verbosity=1,
        )

        model.fit(
            np.stack((data_v1_SR, data_v2_SR, data_z_SR), axis=1),
            data_z_dot_SR,
            variable_names=["v1", "v2", "z"],
        )

        print(time.time() - time0, "s")

        variables = ["v1", "v2", "z"]
        best_expr, best_score = select_expression(model, variables)
        print(f"Best expression after filtering: {best_expr}, score={best_score}")

        z_dot_expr = best_expr

        z_dot_pred = evaluate_expression(
            z_dot_expr, variables, data_v1_SR, data_v2_SR, data_z_SR
        )

        mse_z_dot = mean_squared_error(z_dot_pred, data_z_dot_SR)
        print(f"PySR Mean Squared Error for z_dot: {mse_z_dot:.10f}")

        print("Step 2 parameters:")
        print(f"m1 = {m1_file:.6f}, m2 = {m2_file:.6f}")
        print(f"c11 = {c11_file:.6f}, c12 = {c12_file:.6f}")
        print(f"c21 = {c21_file:.6f}, c22 = {c22_file:.6f}")
        print(f"k11 = {k11_file:.6f}, k12 = {k12_file:.6f}")
        print(f"k21 = {k21_file:.6f}, k22 = {k22_file:.6f}")
        print(f"h1 = {h1_file:.6f}, h2 = {h2_file:.6f}")

        print("Step 3: Learned expression z_dot(v1, v2, z) =", z_dot_expr)
        print("True expression: z_dot = 50.0(v2 - v1) - 2|v2 - v1|z")

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
            data_t_SR=data_t_SR,
            data_x1_SR=data_x1_SR,
            data_x2_SR=data_x2_SR,
            data_v1_SR=data_v1_SR,
            data_v2_SR=data_v2_SR,
            data_z_SR=data_z_SR,
            data_z_dot_SR=data_z_dot_SR,
            z_dot_pred=z_dot_pred,
        )

        session.save_equations(z_dot=z_dot_expr)

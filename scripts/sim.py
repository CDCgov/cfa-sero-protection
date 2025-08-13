# %% Import modules
import altair as alt
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
from jax import random
from numpyro.infer import MCMC, NUTS, init_to_sample
from scipy.optimize import minimize
from scipy.stats import loguniform

# %% Set parameters and establish functions to recreate Casey's results
N = 10000
N_PLOT = 1000
CONTROLS_PER_CASE = 1
SIGMOID_MAX = 0.5
SIGMOID_SLOPE = 1.0
SIGMOID_MID = 4.0
SIGMOID_PARS = {
    "steepness": SIGMOID_SLOPE,
    "mid_point": SIGMOID_MID,
    "saturating_point": SIGMOID_MAX,
}


def generate_paired_samples(sz, list1, list2):
    """Generate a paired random sample (with replacement) of size 'sz' from two lists
    Returns two lists containing matched samples from list1 and list2
    """
    list1_samples = []
    list2_samples = []
    list_size = len(list1)
    for ii in range(sz):
        to_sample = np.random.randint(0, list_size)
        list1_samples.append(list1[to_sample])
        list2_samples.append(list2[to_sample])
    return np.array(list1_samples), np.array(list2_samples)


def jitter_vector(vector, magnitude):
    return vector + np.random.uniform(-magnitude, magnitude, size=len(vector))


def get_loguniform_Ab_titers(size: int) -> np.ndarray:
    # Generate uniform random sample in log-space of antibody titers between ~[0,18000]
    return np.log(loguniform.rvs(np.exp(1), np.exp(10), size=size))


def sigmoid(x, steepness, mid_point, saturating_point=1.0):
    return saturating_point / (1 + np.exp(steepness * (x - mid_point)))


def one_minus_OR(risk):
    baseline_odds = np.max(risk) / (1 - np.max(risk))
    return 1 - ((risk / (1 - risk)) / baseline_odds)


def generate_TND_data(
    N,
    steepness,
    mid_point,
    saturating_point,
    controls_per_case,
):
    """
    Generate test-negative design data. Returns samples of [x, y],
    where x = sampled antibody titer, and y = positive or negative (0/1)
    """
    Ab_titers = np.array([])
    test_results = np.array([])

    # Set constants
    num_desired_cases = int(N / (controls_per_case + 1))
    num_desired_controls = int(N - num_desired_cases)
    pop_to_simulate = N * 10

    # Generate simulated population with antibody titers
    pop_Abs = get_loguniform_Ab_titers(pop_to_simulate)
    risk = sigmoid(pop_Abs, steepness, mid_point, saturating_point)

    # Generate controls
    controls = np.random.choice(pop_Abs, num_desired_controls, replace=False)
    Ab_titers = np.concatenate((Ab_titers, controls))
    test_results = np.concatenate(
        (test_results, np.zeros(num_desired_controls))
    )

    # Generate cases
    cases = np.random.choice(
        pop_Abs, num_desired_cases, replace=False, p=(risk / np.sum(risk))
    )
    Ab_titers = np.concatenate((Ab_titers, cases))
    test_results = np.concatenate((test_results, np.ones(num_desired_cases)))

    return [Ab_titers, test_results]


# Define the scaled logit model
def scaled_logit(x, k, beta_0, beta_1):
    """
    Scaled logistic function

    Parameters:
    x (array-like): Input values.
    k (float): Maximum value (scale).
    beta_0 (float): Intercept parameter for linear regression
    beta_1 (float): Slope parameter.

    Returns:
    array-like: Scaled logistic function values.
    """
    return k / (1 + np.exp(beta_0 + beta_1 * x))


def neg_log_likelihood_scaled_logit(params, data, lambda_reg=0.1):
    # unpack data
    k, beta_0, beta_1 = params
    Abs = np.array(data[0])
    infected = np.array(data[1])

    # Get likelihood
    epsilon = 1e-10  # To prevent log(0)
    prob_pos = infected * np.log(
        np.clip(scaled_logit(Abs, k, beta_0, beta_1), epsilon, 1 - epsilon)
    )
    prob_neg = (1 - infected) * np.log(
        np.clip(1 - scaled_logit(Abs, k, beta_0, beta_1), epsilon, 1 - epsilon)
    )

    # Add L2 regularization penalty on beta_1 to discourage steep slopes
    regularization_penalty = lambda_reg * beta_1**2

    return -1 * np.sum(prob_pos + prob_neg) + regularization_penalty


# Modified function to fit the scaled logit model with regularization
def fit_scaled_logit(
    x_data, y_data, lambda_reg=0.1, initial_guess=(0.5, -1, 1)
):
    """
    Fit the scaled logistic regression model with regularization to avoid steep slopes.

    Parameters:
    x_data (array-like): Independent variable values.
    y_data (array-like): Dependent variable values.
    lambda_reg (float): Regularization parameter. Higher values penalize steeper slopes more.
    initial_guess (tuple): Initial guesses for k, beta_0, and beta_1.

    Returns:
    tuple: Fitted parameters (k, beta_0, beta_1).
    """
    data = [x_data, y_data]

    result = minimize(
        neg_log_likelihood_scaled_logit,
        initial_guess,
        method="Nelder-Mead",
        args=(data, lambda_reg),
        options={
            "xatol": 1e-10,
            "fatol": 1e-10,
            "maxiter": 10000,
            "maxfev": 20000,
        },
    )

    if not result.success:
        print("Did not find optimal parameter fit to minimize likelihood")
    return result.x


# %% Simulate TND data, subset for plotting, fit risk function, and convert to protection
TND = generate_TND_data(
    N, SIGMOID_SLOPE, SIGMOID_MID, SIGMOID_MAX, CONTROLS_PER_CASE
)
Ab_logged = TND[0]
infected = TND[1]
Ab_logged_plot, infected_plot = generate_paired_samples(
    N_PLOT, Ab_logged, infected
)
infected_plot = jitter_vector(infected_plot, 0.1)
potential_Ab_logged = np.arange(min(Ab_logged), max(Ab_logged), 0.01)
true_risk = sigmoid(
    potential_Ab_logged, SIGMOID_SLOPE, SIGMOID_MID, SIGMOID_MAX
)
true_protection = 1 - true_risk / SIGMOID_MAX

# Fit scaled logistic model
fitted_params = fit_scaled_logit(Ab_logged, infected)
fitted_k, fitted_beta_0, fitted_beta_1 = fitted_params
fitted_risk = scaled_logit(
    potential_Ab_logged, fitted_k, fitted_beta_0, fitted_beta_1
)
fitted_protection = one_minus_OR(fitted_risk)

# %% Plot TND data, true risk curve, and frequentist fitted risk curve
plot = (
    (
        alt.Chart(
            pl.DataFrame({"titer": Ab_logged_plot, "status": infected_plot})
        )
        .mark_point(opacity=0.1, color="black")
        .encode(x="titer:Q", y="status:Q")
    )
    + alt.Chart(
        pl.DataFrame({"titer": potential_Ab_logged, "risk": fitted_risk})
    )
    .mark_line(color="green", strokeWidth=5)
    .encode(
        x=alt.X(
            "titer:Q",
            title="Titer",
            axis=alt.Axis(grid=False),
        ),
        y=alt.Y("risk:Q", title="Risk", axis=alt.Axis(grid=False)),
    )
    + alt.Chart(
        pl.DataFrame({"titer": potential_Ab_logged, "risk": true_risk})
    )
    .mark_line(color="black", strokeDash=[3, 3])
    .encode(x="titer:Q", y="risk:Q")
)
plot.display()

# %% Plot TND data, true protection curve, and frequentist fitted protection curve
plot = (
    (
        alt.Chart(
            pl.DataFrame({"titer": Ab_logged_plot, "status": infected_plot})
        )
        .mark_point(opacity=0.1, color="black")
        .encode(x="titer:Q", y="status:Q")
    )
    + alt.Chart(
        pl.DataFrame(
            {"titer": potential_Ab_logged, "protection": fitted_protection}
        )
    )
    .mark_line(color="green", strokeWidth=5)
    .encode(
        x=alt.X(
            "titer:Q",
            title="Titer",
            scale=alt.Scale(padding=0),
            axis=alt.Axis(grid=False),
        ),
        y=alt.Y("protection:Q", title="Protection", axis=alt.Axis(grid=False)),
    )
    + alt.Chart(
        pl.DataFrame(
            {"titer": potential_Ab_logged, "protection": true_protection}
        )
    )
    .mark_line(color="black", strokeDash=[3, 3])
    .encode(x="titer:Q", y="protection:Q")
)
plot.display()


# %% Build a numpyro model of protection
def fit_scaled_logit_bayesian(
    titer,
    infected,
    slope_shape=8.0,
    slope_rate=4.0,
    midpoint_shape=20.0,
    midpoint_rate=4.0,
    max_risk_shape1=5.0,
    max_risk_shape2=2.0,
):
    slope = numpyro.sample("slope", dist.Gamma(slope_shape, slope_rate))
    midpoint = numpyro.sample(
        "midpoint", dist.Gamma(midpoint_shape, midpoint_rate)
    )
    max_risk = numpyro.sample(
        "max_risk", dist.Beta(max_risk_shape1, max_risk_shape2)
    )
    mu = max_risk / (1 + jnp.exp(slope * (titer - midpoint)))
    numpyro.sample("obs", dist.Binomial(probs=mu), obs=infected)


# %% Fit the numpyro model of protection
NUM_SAMPLES = 200
kernel = NUTS(fit_scaled_logit_bayesian, init_strategy=init_to_sample)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=NUM_SAMPLES, num_chains=4)
mcmc.run(random.key(0), titer=Ab_logged, infected=infected)
mcmc.print_summary()

# %% Prepare bayesian posterior samples for plotting
mcmc_samples = mcmc.get_samples()
prot = []
for i in range(NUM_SAMPLES):
    new_prot_infer = (
        pl.DataFrame({"titer": potential_Ab_logged})
        .with_columns(
            risk=sigmoid(
                pl.col("titer"),
                mcmc_samples["slope"][i],
                mcmc_samples["midpoint"][i],
                mcmc_samples["max_risk"][i],
            ),
            sample_id=pl.lit(i),
        )
        .with_columns(
            protection=1
            - (pl.col("risk") / (1 - pl.col("risk")))
            / (pl.col("risk").max() / (1 - pl.col("risk").max()))
        )
    )
    prot.append(new_prot_infer)
prot_infer = pl.concat(prot)

# %% Plot TND data, true risk curve, and bayesian fitted risk curve
alt.data_transformers.disable_max_rows()
plot = (
    (
        alt.Chart(
            pl.DataFrame({"titer": Ab_logged_plot, "status": infected_plot})
        )
        .mark_point(opacity=0.1, color="black")
        .encode(x="titer:Q", y="status:Q")
    )
    + alt.Chart(pl.DataFrame(prot_infer))
    .mark_line(color="green", opacity=2 / NUM_SAMPLES)
    .encode(
        x=alt.X(
            "titer:Q",
            title="Titer",
            axis=alt.Axis(grid=False),
        ),
        y=alt.Y("risk:Q", title="Risk", axis=alt.Axis(grid=False)),
        detail="sample_id",
    )
    + alt.Chart(
        pl.DataFrame({"titer": potential_Ab_logged, "risk": true_risk})
    )
    .mark_line(color="black", strokeDash=[3, 3])
    .encode(x="titer:Q", y="risk:Q")
)
plot.display()

# %% Plot TND data, true protection curve, and bayesian fitted protection curve
alt.data_transformers.disable_max_rows()
plot = (
    (
        alt.Chart(
            pl.DataFrame({"titer": Ab_logged_plot, "status": infected_plot})
        )
        .mark_point(opacity=0.1, color="black")
        .encode(x="titer:Q", y="status:Q")
    )
    + alt.Chart(pl.DataFrame(prot_infer))
    .mark_line(color="green", opacity=2 / NUM_SAMPLES)
    .encode(
        x=alt.X(
            "titer:Q",
            title="Titer",
            axis=alt.Axis(grid=False),
        ),
        y=alt.Y("protection:Q", title="Protection", axis=alt.Axis(grid=False)),
        detail="sample_id",
    )
    + alt.Chart(
        pl.DataFrame(
            {"titer": potential_Ab_logged, "protection": true_protection}
        )
    )
    .mark_line(color="black", strokeDash=[3, 3])
    .encode(x="titer:Q", y="protection:Q")
)
plot.display()

# %% Define parameters
POP_SIZE = 10000
NUM_DAYS = 100
AB_RISK_SLOPE = 2
AB_RISK_MIDPOINT = 6
AB_RISK_MAX = 0.75
AB_RISK_MIDPOINT_DROP = 0.0
AB_DECAY = [0.9, 1.0]
AB_SPIKE = [8, 10]
NONAB_BIAS = 2
NONAB_NOISE = 1
NONAB_RISK_SLOPE = 2
NONAB_RISK_MIDPOINT = 6
NONAB_RISK_MAX = 0.75
NONAB_RISK_MIDPOINT_DROP = 0.0
NONAB_DECAY = [0.9, 1.0]
LAG = 4
FORCE_EXP = 0.1
FORCE_VAX = 0.01
RECOVERY = 0.25
XSEC_SIZE = 100
XSEC_WINDOW = [50, 57]
TND_INF_PRB = 0.1
TND_NON_PRB = 0.01


# %% Define functions
def calculate_risk(
    titer: pl.Expr,
    slope: float,
    midpoint: float,
    max_risk: float,
) -> pl.Expr:
    """
    Define risk as a function of log antibody titer,
    following a scaled logit functional form.
    """

    return max_risk / (1 + (slope * (titer - midpoint)).exp())


def calculate_oddsratio(risk: pl.Expr) -> pl.Expr:
    """
    Define protection as one minus the odds ratio.
    """
    return 1 - ((risk / (1 - risk)) / (risk.max() / (1 - risk.max())))


# %% Run simulation
daily_data = []

for i in range(NUM_DAYS):
    if i == 0:
        new_daily_data = pl.DataFrame(
            {
                "id": range(0, POP_SIZE),
                "inf_status": [False] * POP_SIZE,
                "vax_status": [False] * POP_SIZE,
                "ab": [0.0] * POP_SIZE,
                "nonab": [0.0] * POP_SIZE,
                "inf_new_lag": [False] * POP_SIZE,
                "vax_new_lag": [False] * POP_SIZE,
            }
        )
    else:
        new_daily_data = (
            daily_data[i - 1]
            .with_columns(
                ab=pl.col("ab")
                * pl.Series(
                    "ab", np.random.uniform(AB_DECAY[0], AB_DECAY[1], POP_SIZE)
                )
            )
            .with_columns(
                nonab=pl.col("nonab")
                * pl.Series(
                    np.random.uniform(NONAB_DECAY[0], NONAB_DECAY[1], POP_SIZE)
                )
            )
            .with_columns(
                inf_new_lag=daily_data[max(0, i - LAG)]["inf_new"],
                vax_new_lag=daily_data[max(0, i - LAG)]["vax_new"],
            )
            .with_columns(
                ab=pl.when(pl.col("inf_new_lag") | pl.col("vax_new_lag"))
                .then(
                    pl.Series(
                        np.random.uniform(AB_SPIKE[0], AB_SPIKE[1], POP_SIZE),
                    )
                )
                .otherwise(pl.col("ab"))
            )
            .with_columns(
                nonab=pl.when(pl.col("inf_new_lag") | pl.col("vax_new_lag"))
                .then(
                    pl.col("ab")
                    + pl.Series(
                        np.random.normal(NONAB_BIAS, NONAB_NOISE, POP_SIZE),
                    )
                )
                .otherwise(pl.col("nonab")),
            )
        )

    new_daily_data = (
        new_daily_data.with_columns(
            ab_risk=calculate_risk(
                pl.col("ab"),
                AB_RISK_SLOPE,
                AB_RISK_MIDPOINT - (AB_RISK_MIDPOINT_DROP * i),
                AB_RISK_MAX,
            ),
            nonab_risk=calculate_risk(
                pl.col("nonab"),
                NONAB_RISK_SLOPE,
                NONAB_RISK_MIDPOINT - (NONAB_RISK_MIDPOINT_DROP * i),
                NONAB_RISK_MAX,
            ),
            inf_draw=pl.Series("inf_draw", np.random.rand(POP_SIZE))
            / FORCE_EXP,
            vax_draw=pl.Series("vax_draw", np.random.rand(POP_SIZE))
            / FORCE_VAX,
            rec_draw=pl.Series("rec_draw", np.random.rand(POP_SIZE))
            / RECOVERY,
            day=i,
        )
        .with_columns(
            inf_new=pl.when(
                (
                    pl.col("inf_draw")
                    < pl.min_horizontal("ab_risk", "nonab_risk")
                )
                & ~pl.col("inf_status")
            )
            .then(True)
            .otherwise(False)
        )
        .with_columns(
            inf_status=pl.col("inf_status") | pl.col("inf_new"),
        )
        .with_columns(
            vax_new=pl.when(
                (pl.col("vax_draw") < 1.0)
                & ~pl.col("inf_status")
                & ~pl.col("vax_status")
            )
            .then(True)
            .otherwise(False)
        )
        .with_columns(vax_status=pl.col("vax_status") | pl.col("vax_new"))
        .with_columns(
            rec_new=pl.when((pl.col("rec_draw") < 1.0) & ~pl.col("inf_new"))
            .then(True)
            .otherwise(False)
        )
        .with_columns(
            inf_status=pl.when(pl.col("rec_new"))
            .then(False)
            .otherwise(pl.col("inf_status"))
        )
    )

    daily_data.append(new_daily_data)

all_data = pl.concat(daily_data)

# %% Plot number of people currently infected through time
inf_plot = all_data.group_by("day").agg(pl.col("inf_status").sum())
alt.Chart(inf_plot).mark_line(color="black").encode(
    x=alt.X("day:Q", title="Day"),
    y=alt.Y("inf_status:Q", title="Current Number Infected"),
)

# %% Plot number of people cumulatively vaccinated through time
vax_plot = all_data.group_by("day").agg(pl.col("vax_status").sum())
alt.Chart(vax_plot).mark_line(color="black").encode(
    x=alt.X("day:Q", title="Day"),
    y=alt.Y("vax_status:Q", title="Cumulative Number Vaccinated"),
)

# %% Plot the antibody protection curve
pro_plot = (
    pl.DataFrame({"ab": np.arange(0, AB_SPIKE[1], 0.01)})
    .with_columns(
        risk=calculate_risk(
            pl.col("ab"),
            AB_RISK_SLOPE,
            AB_RISK_MIDPOINT,
            AB_RISK_MAX,
        )
    )
    .with_columns(protection=1 - pl.col("risk") / AB_RISK_MAX)
)
alt.Chart(pro_plot).mark_line(color="black", strokeDash=[3, 3]).encode(
    x=alt.X("ab:Q", title="Antibody Titer"),
    y=alt.Y("risk:Q", title="Risk", scale=alt.Scale(domain=[0, 1])),
)
alt.Chart(pro_plot).mark_line(color="black", strokeDash=[3, 3]).encode(
    x=alt.X("ab:Q", title="Antibody Titer"),
    y=alt.Y(
        "protection:Q", title="Protection", scale=alt.Scale(domain=[0, 1])
    ),
)
# %% Plot antibody dynamics across 100 days for 15 randomly selected individuals
plot_sub_data = all_data.filter(
    pl.col("id").is_in(all_data.select("id").sample(n=5)["id"].to_list())
).with_columns(id=pl.col("id").cast(pl.String))
alt.data_transformers.disable_max_rows()
plot = (
    alt.Chart(plot_sub_data)
    .mark_line(color="black", opacity=0.8)
    .encode(
        x=alt.X(
            "day:Q",
            title="Day",
        ),
        y=alt.Y("ab:Q", title="Antibody Titer"),
        color="id",
    )
)
plot.display()


# %% Plot antibody dynamics for 100 randomly selected exposure events
plot_data = (
    all_data.with_columns(any_new=pl.col("inf_new") | pl.col("vax_new"))
    .with_columns(
        exposure_count=pl.col("any_new").cast(pl.UInt32).cum_sum().over("id")
    )
    .with_columns(
        days_since_exposure=pl.int_range(0, pl.len()).over(
            ["id", "exposure_count"]
        ),
        exposure_id=pl.struct(["id", "exposure_count"]).rank(method="dense"),
    )
    .filter(pl.col("exposure_count") > 0)
)

plot_sub_data = plot_data.filter(
    pl.col("exposure_id").is_in(
        plot_data.select("exposure_id").sample(n=100)["exposure_id"].to_list()
    )
)

alt.data_transformers.disable_max_rows()
plot = (
    alt.Chart(plot_sub_data)
    .mark_line(color="black", opacity=0.1)
    .encode(
        x=alt.X(
            "days_since_exposure:Q",
            title="Days Since Exposure",
        ),
        y=alt.Y("ab:Q", title="Antibody Titer"),
        detail="exposure_id",
    )
)
plot.display()

# %% Plot mean antibody and nonantibody titers through time
plot_data = (
    all_data.group_by("day")
    .agg(
        ab_mean=pl.col("ab").mean(),
        nonab_mean=pl.col("nonab").mean(),
    )
    .unpivot(index="day", variable_name="component", value_name="titer")
    .with_columns(
        component=pl.col("component").replace_strict(
            {"ab_mean": "ab", "nonab_mean": "non_ab"}
        )
    )
)
plot = (
    alt.Chart(plot_data)
    .mark_line()
    .encode(
        x=alt.X(
            "day:Q",
            title="Day",
        ),
        y=alt.Y("titer:Q", title="Population Average Titer"),
        color="component",
    )
)
plot.display()

# %% Plot correlation of nonAb with Ab values
alt.data_transformers.disable_max_rows()
plot = (
    alt.Chart(
        all_data.filter((pl.col("ab") > 0) | (pl.col("nonab") > 0)).sample(
            n=500
        )
    )
    .mark_point(color="black", opacity=0.5)
    .encode(
        x=alt.X(
            "ab:Q",
            title="Antibody Titer",
        ),
        y=alt.Y("nonab:Q", title="Non-Antibody Titer"),
    )
) + alt.Chart(pl.DataFrame({"x": range(11), "y": range(11)})).mark_line(
    color="black", strokeDash=[10, 10]
).encode(x="x:Q", y="y:Q")
plot.display()

# %% Conduct a TND serosurvey
tnd_inf_samples = all_data.filter(pl.col("inf_new")).sample(
    fraction=TND_INF_PRB
)
tnd_non_samples = all_data.filter(~pl.col("inf_status")).sample(
    fraction=TND_NON_PRB
)
tnd_data = pl.concat([tnd_inf_samples, tnd_non_samples])

# %% Plot number of cases and controls sampled each day
plot_data = tnd_data.group_by(["day", "inf_status"]).len()
plot = (
    alt.Chart(plot_data)
    .mark_line()
    .encode(
        x=alt.X(
            "day:Q",
            title="Day",
        ),
        y=alt.Y("len:Q", title="Number Sampled"),
        color="inf_status",
    )
)
plot.display()


# %% Fit the numpyro model of protection
NUM_SAMPLES = 200
titer = tnd_data["ab"].to_numpy()
infected = tnd_data["inf_status"].to_numpy() * 1
kernel = NUTS(fit_scaled_logit_bayesian, init_strategy=init_to_sample)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=NUM_SAMPLES, num_chains=4)
mcmc.run(random.key(0), titer=titer, infected=infected)
mcmc.print_summary()

# %% Prepare posterior samples to plot true risk/protection vs. inferred
prot_real = (
    pl.DataFrame({"ab": np.arange(0, AB_SPIKE[1], 0.01)})
    .with_columns(
        risk=calculate_risk(
            pl.col("ab"),
            AB_RISK_SLOPE,
            AB_RISK_MIDPOINT,
            AB_RISK_MAX,
        )
    )
    .with_columns(protection=1 - pl.col("risk") / AB_RISK_MAX)
)
mcmc_samples = mcmc.get_samples()
prot = []
for i in range(1000):
    new_prot_infer = (
        pl.DataFrame({"ab": np.arange(0, AB_SPIKE[1], 0.01)})
        .with_columns(
            risk=calculate_risk(
                pl.col("ab"),
                mcmc_samples["slope"][i],
                mcmc_samples["midpoint"][i],
                mcmc_samples["max_risk"][i],
            ),
            sample_id=pl.lit(i),
        )
        .with_columns(protection=calculate_oddsratio(pl.col("risk")))
    )
    prot.append(new_prot_infer)
prot_infer = pl.concat(prot)

# %% Plot true risk vs. inferred
alt.data_transformers.disable_max_rows()
output = (
    (
        alt.Chart(
            tnd_data.with_columns(
                inf_status=pl.col("inf_status") * 1
                + np.random.uniform(-0.1, 0.1, tnd_data.height)
            ).sample(n=1000)
        )
        .mark_point(opacity=0.1, color="black")
        .encode(x="ab:Q", y="inf_status:Q")
    )
    + alt.Chart(prot_infer)
    .mark_line(opacity=2 / NUM_SAMPLES, color="green")
    .encode(
        x=alt.X("ab:Q", title="Antibody Titer"),
        y=alt.Y("risk:Q", title="Risk"),
        detail="sample_id",
    )
    + alt.Chart(prot_real)
    .mark_line(color="black", strokeDash=[3, 3])
    .encode(
        x=alt.X("ab:Q", title="Antibody Titer"),
        y=alt.Y("risk:Q", title="Risk"),
    )
)
output.display()

# %% Plot true protection vs. inferred
alt.data_transformers.disable_max_rows()
output = (
    (
        alt.Chart(
            tnd_data.with_columns(
                inf_status=pl.col("inf_status") * 1
                + np.random.uniform(-0.1, 0.1, tnd_data.height)
            ).sample(n=1000)
        )
        .mark_point(opacity=0.1, color="black")
        .encode(x="ab:Q", y="inf_status:Q")
    )
    + alt.Chart(prot_infer)
    .mark_line(opacity=2 / NUM_SAMPLES, color="green")
    .encode(
        x=alt.X("ab:Q", title="Antibody Titer"),
        y=alt.Y("protection:Q", title="Protection"),
        detail="sample_id",
    )
    + alt.Chart(prot_real)
    .mark_line(color="black", strokeDash=[3, 3])
    .encode(
        x=alt.X("ab:Q", title="Antibody Titer"),
        y=alt.Y("protection:Q", title="Protection"),
    )
)
output.display()

# %%

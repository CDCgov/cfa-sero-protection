# %% Import my modules
import sys

sys.path.append("/home/tec0/cfa-sero-protection")
import seropro.samples as sps
import seropro.utils as spu

# %% Simulate titers and plot distribution
titers = sps.simulate_titers(
    [("gamma", 1, 100, 100), ("gamma", 2, 100, 100), ("gamma", 3, 100, 100)]
)
titers.to_density(values="titer", groups=None).plot()

# %% Simulate risk curve posterior distribution and plot
curves = sps.simulate_curve(
    {
        "mid": ("normal", 120, 10, 100),
        "slope": ("beta", 2, 100, 100),
        "min": ("beta", 5, 100, 100),
        "max": ("beta", 500, 100, 100),
    }
)
curves.plot(spu.calculate_risk_dslogit)

# %% Calculate risks from titers and plot their distribution
risks = titers.to_risk(curves, spu.calculate_risk_dslogit)
risks.to_density(values="risk", groups="par_id").plot()

# %% Calculate odds-ratio protections for risks and plot their distribution
prots = risks.to_protection(spu.calculate_protection_oddsratio)
prots.to_density(values="protection", groups="par_id").plot()

# %% Calculate risk-ratio protections for risks and plot their distribution
prots = risks.to_protection(spu.calculate_protection_riskratio)
prots.to_density(values="protection", groups="par_id").plot()

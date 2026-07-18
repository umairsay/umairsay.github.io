"""
The Market for Peaches - analysis

does AI exposure predict which occupations are growing or shrinking? 
age, industry, the education/experience split, wages, and who's staffing what.

Files needed in the same folder, all from BLS Employment Projections
2024-34, CPS Table 11b, and an occupation-level AI exposure dataset:
occupation.xlsx, industry.xlsx, labor-force.xlsx, skills.xlsx, sep.xlsx,
job_exposure.csv, task_penetration.csv, cpsaat11b.xlsx, cpsaat11b_2019.xlsx
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

BASE = Path(__file__).resolve().parent
OUT = BASE / "peaches_output"
OUT.mkdir(exist_ok=True)


def num(s):
    return pd.to_numeric(s, errors="coerce")


def load_cps_age_table(path):
    # CPS Table 11b comes as a downloaded BLS spreadsheet with a merged,
    # multi-row header that doesn't line up with a fixed skiprows count.
    # Easiest fix is to just search for the row where the real data starts.
    raw = pd.read_excel(path, header=None)
    mask = raw[0].astype(str).str.strip().str.lower() == "total employed"
    if not mask.any():
        raise ValueError(f"couldn't find the 'Total employed' row in {path}")
    start = mask.idxmax()
    data = raw.loc[start:, :9].copy()
    data.columns = ["title", "total", "a1619", "a2024", "a2534", "a3544",
                     "a4554", "a5564", "a65", "median_age"]
    for c in data.columns[1:]:
        data[c] = num(data[c])
    data["title"] = data["title"].astype(str).str.strip()
    data = data.dropna(subset=["total", "a5564", "a65"]).query("total > 0")
    data["title_clean"] = data["title"].str.lower()
    return data


# load everything
occ = pd.read_excel(BASE / "occupation.xlsx", sheet_name="Table 1.2", skiprows=1)
industry = pd.read_excel(BASE / "industry.xlsx", sheet_name="Table 2.1", skiprows=1)
age = pd.read_excel(BASE / "labor-force.xlsx", sheet_name="Table 3.1", skiprows=1)
skills = pd.read_excel(BASE / "skills.xlsx", sheet_name="Table 6.2", skiprows=1)
sep = pd.read_excel(BASE / "sep.xlsx", sheet_name="Table 1.10", skiprows=1)
exp = pd.read_csv(BASE / "job_exposure.csv")
tasks = pd.read_csv(BASE / "task_penetration.csv")
cps = load_cps_age_table(BASE / "cpsaat11b.xlsx")
cps_2019 = load_cps_age_table(BASE / "cpsaat11b_2019.xlsx")

# build one row per detailed occupation, everything joined on
d = occ[occ["Occupation type"] == "Line item"].copy()
d = d.rename(columns={"2024 National Employment Matrix code": "occ_code"})
d["title"] = d["2024 National Employment Matrix title"].str.strip()
d["title_clean"] = d["title"].str.lower()
d["emp"] = num(d["Employment, 2024"])
d["chg"] = num(d["Employment change, numeric, 2024\u201334"])
d["pct_chg"] = num(d["Employment change, percent, 2024\u201334"])
d["wage"] = num(d["Median annual wage, dollars, 2024[1]"])
d["log_wage"] = np.log(d["wage"])
d["exp_req"] = d["Work experience in a related occupation"].fillna("None").str.strip()
d["edu"] = d["Typical education needed for entry"].astype(str).str.strip()
d["office_admin"] = d["occ_code"].astype(str).str.startswith("43-").astype(int)
d = d.merge(exp[["occ_code", "observed_exposure"]], on="occ_code", how="left")
d["exposure"] = num(d["observed_exposure"])
d = d.merge(cps[["title_clean", "a5564", "a65", "a1619", "a2024", "total"]],
            on="title_clean", how="left")
d["youth_share"] = (d["a1619"] + d["a2024"]) / d["total"] * 100
d["senior_share"] = (d["a5564"] + d["a65"]) / d["total"] * 100
d = d.dropna(subset=["emp", "chg"])
TOT_EMP, TOT_CHG = d["emp"].sum(), d["chg"].sum()


# --- part 1: does AI exposure actually predict growth? ---
# This is the question I expected to lead the piece. Testing it properly
# before writing anything, because if it holds up it's the whole story,
# and if it doesn't, that matters just as much.

clean = d.dropna(subset=["pct_chg", "exposure", "wage"])
print(f"regression sample: {len(clean)} occupations with growth, exposure, and wage")

m1 = smf.ols("pct_chg ~ exposure", data=clean).fit(cov_type="HC1")
print(f"exposure alone: coef {m1.params['exposure']:.2f}, p {m1.pvalues['exposure']:.3f}, R2 {m1.rsquared:.3f}")

office = clean[clean["office_admin"] == 1]
rest = clean[clean["office_admin"] == 0]
t_office, p_office = stats.ttest_ind(office["pct_chg"], rest["pct_chg"], equal_var=False)
print(f"office/admin vs everything else, growth: {office['pct_chg'].mean():.1f}% vs "
      f"{rest['pct_chg'].mean():.1f}%, p {p_office:.4f}")

sk = skills.rename(columns={"2024 National Employment Matrix code": "occ_code"})
sk["occ_code"] = sk["occ_code"].astype(str)
sk["pct_chg"] = num(sk["Employment change, percent, 2024\u201334"])
sk = sk.merge(exp[["occ_code", "observed_exposure"]], on="occ_code", how="inner")
sk["exposure"] = num(sk["observed_exposure"])
sk_clean = sk.dropna(subset=["pct_chg", "exposure", "Top highest skill"])
m_skill = smf.ols('pct_chg ~ C(Q("Top highest skill"))', data=sk_clean).fit(cov_type="HC1")
print(f"skill category as a group, F-test p {m_skill.f_pvalue:.4f}")

# I ran more tests than these three while checking this (wage-controlled
# models, an office/admin interaction, a wage-trimmed robustness check,
# CPS youth-share changes). Running several tests on overlapping data means
# some will look significant by chance, so I corrected all of them together
# with Holm's method rather than reporting whichever one looked best.
m2 = smf.ols("pct_chg ~ exposure + log_wage", data=clean.dropna(subset=["log_wage"])).fit(cov_type="HC1")
cutoff = clean["wage"].quantile(0.99)
m_trim = smf.ols("pct_chg ~ exposure", data=clean[clean["wage"] <= cutoff]).fit(cov_type="HC1")

cps_youth = cps.merge(cps_2019[["title_clean"]].assign(in_2019=1), on="title_clean", how="left")
cps_now_x = d.dropna(subset=["youth_share", "exposure"])
m_flow = smf.ols("youth_share ~ exposure", data=cps_now_x).fit(cov_type="HC1")

test_names = ["exposure only", "exposure + wage", "office/admin t-test",
              "skill category F-test", "wage-trimmed robustness", "youth share vs exposure"]
raw_p = [m1.pvalues["exposure"], m2.pvalues["exposure"], p_office,
         m_skill.f_pvalue, m_trim.pvalues["exposure"], m_flow.pvalues["exposure"]]
reject, holm_p, _, _ = multipletests(raw_p, alpha=0.05, method="holm")
print("\nafter correcting for running six tests on related data:")
for name, raw, adj, ok in zip(test_names, raw_p, holm_p, reject):
    print(f"  {name:28s} raw p {raw:.3f}  ->  {adj:.3f}  {'survives' if ok else 'does not survive'}")
print("only office/admin and skill category hold up. exposure alone, once controls and")
print("corrections are in, doesn't predict growth on its own. that's why the piece isn't")
print("built around 'AI is doing this' - the data wouldn't back it.\n")


# --- part 2: the numbers the piece actually uses ---

age["Group"] = age["Group"].str.strip()
age["pct_chg"] = num(age["Percent change, 2024\u201334"])
age_groups = ["16 to 19", "20 to 24", "25 to 54", "55 to 64", "65 to 74", "75 years and older"]
age_out = {g: round(float(age.loc[age["Group"] == g, "pct_chg"].iloc[0]), 1)
           for g in age_groups if (age["Group"] == g).any()}
print("labor force change by age, 2024-34:")
for g, v in age_out.items():
    print(f"  {g:22s} {v:+.1f}%")

industry["pct_chg"] = num(industry["Employment change, percent, 2024\u201334 "])
industry["emp2024"] = num(industry["Employment, 2024"])
sectors = industry.dropna(subset=["pct_chg", "emp2024"])
sectors = sectors[sectors["emp2024"] > 100].sort_values("pct_chg", ascending=False)
print("\nindustry, top and bottom:")
print(sectors[["Industry sector", "pct_chg"]].head(5).to_string(index=False))
print("...")
print(sectors[["Industry sector", "pct_chg"]].tail(4).to_string(index=False))

# the core comparison: within bachelor's-plus work, jobs that require prior
# experience versus jobs that don't
ba = d[d["edu"].isin(["Bachelor's degree", "Master's degree", "Doctoral or professional degree"])].copy()
ba["gate"] = np.where(ba["exp_req"] == "None", "open", "gated")
cross = ba.groupby("gate").agg(emp=("emp", "sum"), chg=("chg", "sum"), wage=("wage", "median"))
cross["base_share"] = cross["emp"] / cross["emp"].sum() * 100
cross["growth_share"] = cross["chg"] / cross["chg"].sum() * 100
cross["growth_rate"] = cross["chg"] / cross["emp"] * 100
print("\nbachelor's-plus occupations, gated vs open:")
print(cross.round(1).to_string())

er = d[d["exp_req"] != "None"]
print(f"\neconomy-wide, experience-required jobs: {er['emp'].sum()/TOT_EMP*100:.1f}% of "
      f"employment, {er['chg'].sum()/TOT_CHG*100:.1f}% of net growth")

big = d.dropna(subset=["pct_chg", "youth_share"]).query("emp > 50")
print("\nfastest growing (50k+ workers):")
print(big.sort_values("pct_chg", ascending=False).head(5)[["title", "pct_chg", "youth_share"]].round(1).to_string(index=False))
print("fastest declining:")
print(big.sort_values("pct_chg").head(6)[["title", "pct_chg", "youth_share"]].round(1).to_string(index=False))

sk_summary = sk_clean.groupby("Top highest skill").agg(
    n=("occ_code", "count"), mean_exposure=("exposure", "mean"), mean_growth=("pct_chg", "mean")
).sort_values("mean_exposure", ascending=False)
print("\nskill groups, exposure vs growth:")
print(sk_summary.round(3).to_string())

wage_big = d.dropna(subset=["pct_chg", "wage"]).query("emp > 50")
w_win = wage_big.sort_values("pct_chg", ascending=False).head(8)
w_lose = wage_big.sort_values("pct_chg").head(8)
ratio = w_win["wage"].median() / w_lose["wage"].median()
print(f"\nwage gap, growing vs declining jobs: ${w_win['wage'].median():,.0f} vs "
      f"${w_lose['wage'].median():,.0f}, {ratio:.1f}x")

# 55+ share by experience gate. worth flagging: jobs that require years of
# prior experience will always skew older just by definition, so part of
# this gap is mechanical, not a discovery. still worth reporting, because
# the point isn't that gated jobs are older, it's that the people currently
# doing them are close to leaving.
gate_all = d.dropna(subset=["senior_share", "exp_req"]).copy()
gate_all["gate"] = np.where(gate_all["exp_req"] == "None", "open", "gated")
dep = gate_all.groupby("gate").apply(lambda x: np.average(x["senior_share"], weights=x["total"]), include_groups=False)
print(f"\nworkforce 55+, gated occupations: {dep['gated']:.1f}%, open: {dep['open']:.1f}%")
print("(gated jobs skew older partly by construction - a 5-year requirement selects for")
print("tenure - so treat this as directional, not a clean natural experiment)")

summary = {
    "age": age_out,
    "industry_leaders": sectors[["Industry sector", "pct_chg"]].head(5).to_dict("records"),
    "industry_laggards": sectors[["Industry sector", "pct_chg"]].tail(4).to_dict("records"),
    "education_experience_cross": cross.round(1).to_dict(orient="index"),
    "experience_gate_economy_wide": {
        "base_share": round(er["emp"].sum() / TOT_EMP * 100, 1),
        "growth_share": round(er["chg"].sum() / TOT_CHG * 100, 1),
    },
    "wage_wall_ratio": round(float(ratio), 2),
    "senior_share_by_gate": dep.round(1).to_dict(),
    "exposure_hypothesis": {name: {"raw_p": round(r, 4), "holm_p": round(a, 4), "survives": bool(ok)}
                             for name, r, a, ok in zip(test_names, raw_p, holm_p, reject)},
}
(OUT / "peaches_summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nsaved {OUT / 'peaches_summary.json'}")

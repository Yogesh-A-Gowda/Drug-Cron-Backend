"""
Scraping sources:
  - OpenFDA (official, free, no key)
  - Site-restricted Google search snippets via SerpAPI (drugs.com, askapatient.com, webmd.com)
  - Cached one-time base datasets (Kaggle UCI + WebMD), pulled from your HF dataset
    repo — populated once via scripts/fetch_kaggle_base.py, not re-downloaded here.

Reddit removed: Reddit closed unauthenticated .json access in May 2026, and
its official API requires OAuth credentials — not worth the added integration
for the marginal data it contributed here.
"""
import os, re, time, requests, pandas as pd
from huggingface_hub import snapshot_download

DRUGS = ["metformin", "sertraline", "gabapentin", "amitriptyline", "lithium", "adderall", "oxycodone"]
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "yogeshagowda/mtech")

SERPAPI_SITES = [
    ("drugs_com", "drugs.com/comments/", "(year-old OR I am OR weight OR lbs OR kg)"),
    ("askapatient", "askapatient.com/viewrating.asp", "(year-old OR I am OR weight OR lbs OR kg)"),
    ("webmd_reviews", "webmd.com/drugs/drugreview", "(year-old OR I am OR weight OR lbs OR kg)"),
]

def extract_demographics(text):
    if pd.isna(text) or not isinstance(text, str):
        return None, None, None
    age = gender = weight = None
    junk_patterns = ["acne and skin issues", "get support and advice", "this page requires javascript"]
    if any(p in text.lower() for p in junk_patterns):
        return None, None, None
    age_match = re.search(r'\b(\d{1,3})[-\s]year[-\s]old|I am (\d{1,3})|\b(\d{1,3})[FfMm]', text, re.IGNORECASE)
    if age_match:
        age = int(next(filter(None, age_match.groups())))
    gender_match = re.search(r'\b(male|female|man|woman|M|F)\b', text, re.IGNORECASE)
    if gender_match:
        g = gender_match.group(1).upper()
        gender = 'male' if g in ['MALE', 'MAN', 'M'] else 'female'
    weight_match = re.search(r'\b(\d{2,3})\s*(lbs|kg)\b', text, re.IGNORECASE)
    if weight_match:
        val = float(weight_match.group(1))
        weight = val if 'kg' in weight_match.group(2).lower() else round(val * 0.453592, 1)
    return age, gender, weight

def scrape_via_serpapi():
    """Sweeps every site in SERPAPI_SITES for every drug, using one shared key."""
    if not SERPAPI_KEY:
        print("SERPAPI_KEY not set — skipping all SerpAPI-based sources")
        return pd.DataFrame()

    records = []
    for source_label, site_filter, extra_terms in SERPAPI_SITES:
        for drug in DRUGS:
            params = {
                "engine": "google",
                "q": f"site:{site_filter} {drug} {extra_terms}",
                "num": 10,
                "api_key": SERPAPI_KEY,
            }
            try:
                r = requests.get("https://serpapi.com/search", params=params, timeout=10)
                if r.status_code == 200:
                    for res in r.json().get("organic_results", []):
                        snippet, title = res.get("snippet", ""), res.get("title", "")
                        age, gender, weight = extract_demographics(f"{title} {snippet}")
                        records.append({
                            "source": source_label,
                            "drug_name": drug,
                            "review_text": snippet,
                            "age": age,
                            "gender": gender,
                            "weight": weight,
                        })
                elif r.status_code == 429:
                    print(f"SerpAPI rate/quota limit hit on {source_label}/{drug} — stopping this source early")
                    break
                time.sleep(1)
            except Exception as e:
                print(f"SerpAPI error on {source_label}/{drug}: {e}")
        print(f"Completed sweep for {source_label}")
    return pd.DataFrame(records)

def scrape_openfda():
    records = []
    for drug in DRUGS:
        try:
            r = requests.get(
                "https://api.fda.gov/drug/event.json",
                params={"search": f"patient.drug.openfda.generic_name:{drug}", "limit": 20},
                headers={"User-Agent": "DrugModelBot/1.0"},
                timeout=10,
            )
            if r.status_code == 200:
                for report in r.json().get("results", []):
                    patient = report.get("patient", {})
                    age, sex, weight = patient.get("patientage"), patient.get("patientsex"), patient.get("patientweight")
                    gender = "male" if sex == "1" else "female" if sex == "2" else None
                    reactions = "; ".join(r_.get("reactionmeddrapt", "") for r_ in patient.get("reaction", [])[:2])
                    drug_name = patient.get("drug", [{}])[0].get("openfda", {}).get("generic_name", ["Unknown"])[0]
                    records.append({
                        "source": "openfda",
                        "drug_name": drug_name,
                        "review_text": reactions,
                        "age": float(age) if age else None,
                        "gender": gender,
                        "weight": float(weight) if weight else None,
                    })
            time.sleep(1)
        except Exception as e:
            print(f"OpenFDA error: {e}")
    return pd.DataFrame(records)


def parse_age_bucket(age_str):
    """WebMD gives ages as ranges ('25-34') or open-ended buckets ('75 or over'),
    not exact values. This approximates a numeric age via midpoint (or the lower
    bound for open-ended buckets) — good enough for imputation math downstream,
    but it's worth knowing this source's ages are approximate, not exact."""
    if pd.isna(age_str):
        return None
    s = str(age_str).strip().lower()
    nums = [float(n) for n in re.findall(r'\d+', s)]
    if not nums:
        return None
    if 'or over' in s or 'or older' in s or '+' in s:
        return nums[0]
    if len(nums) == 2:
        return (nums[0] + nums[1]) / 2
    return nums[0]

def load_cached_base_datasets():
    """
    Pulls the one-time-cached Kaggle base data from your HF dataset repo.
    Populated once via scripts/fetch_kaggle_base.py — this function never
    talks to Kaggle directly, so no Kaggle credentials are needed in the
    recurring pipeline.
    """
    try:
        cache_dir = snapshot_download(repo_id=HF_DATASET_REPO, repo_type="dataset",
                                       token=os.environ.get("HF_TOKEN"))
    except Exception as e:
        print(f"Could not pull base datasets from {HF_DATASET_REPO}: {e}")
        return pd.DataFrame()

    frames = []
    uci_path = os.path.join(cache_dir, "drugsComTrain_raw.csv")
    if os.path.exists(uci_path):
        df = pd.read_csv(uci_path)
        df = df.rename(columns={"drugName": "drug_name", "review": "review_text"})
        df = df[["drug_name", "review_text"]].dropna(subset=["review_text"])
        df["drug_name"] = df["drug_name"].str.lower()
        df[["age", "gender", "weight"]] = df["review_text"].apply(lambda x: pd.Series(extract_demographics(x)))
        df["source"] = "kaggle_uci"
        frames.append(df)
    else:
        print(f"No drugsComTrain_raw.csv found in {HF_DATASET_REPO} — skipping UCI source")

    webmd_path = os.path.join(cache_dir, "webmd.csv")
    if os.path.exists(webmd_path):
        df = pd.read_csv(webmd_path)
        df = df.rename(columns={"Drug": "drug_name", "Reviews": "review_text",
                                 "Age": "age", "Sex": "gender"})
        df = df[["drug_name", "review_text", "age", "gender"]].dropna(subset=["review_text"])
        df["drug_name"] = df["drug_name"].str.lower()
        df["age"] = df["age"].apply(parse_age_bucket)
        df["weight"] = None  # WebMD has no weight column
        df["source"] = "webmd_kaggle"
        frames.append(df)
    else:
        print(f"No webmd.csv found in {HF_DATASET_REPO} — skipping WebMD source")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def run(output_path="drug_reviews_clean.csv"):
    df_serp = scrape_via_serpapi()
    df_fda = scrape_openfda()
    df_base = load_cached_base_datasets()

    df_final = pd.concat([df_serp, df_fda, df_base], ignore_index=True)
    df_final = df_final.dropna(subset=["drug_name", "review_text"], how="all")
    df_final = df_final.drop_duplicates(subset=["review_text"], keep="first")
    df_final = df_final[df_final["age"].isna() | df_final["age"].between(10, 100)]
    df_final = df_final[["source", "drug_name", "review_text", "age", "gender", "weight"]].reset_index(drop=True)

    df_final.to_csv(output_path, index=False)
    print(f"Saved {len(df_final)} reviews to {output_path}")
    return df_final

if __name__ == "__main__":
    run()

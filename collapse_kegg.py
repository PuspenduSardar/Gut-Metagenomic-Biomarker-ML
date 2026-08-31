import argparse
import os
import urllib.request
import pandas as pd


def get_kegg_mapping_table(cache_file="kegg_ko2pathway.tsv"):
    """Downloads and caches the official KO-to-Pathway mapping from KEGG REST API."""
    if os.path.exists(cache_file):
        print(f"Loading cached KEGG pathway map from '{cache_file}'...")
        return pd.read_csv(cache_file, sep="\t")

    print("Fetching KO-to-Pathway mappings from KEGG REST API...")
    url = "https://rest.kegg.jp/link/pathway/ko"

    mappings = []
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        for line in response:
            parts = line.decode("utf-8").strip().split("\t")
            if len(parts) == 2:
                ko = parts[0].replace("ko:", "")
                # Retain numeric pathway IDs (e.g., path:map00010 -> map00010)
                pathway = parts[1].replace("path:", "")
                mappings.append((ko, pathway))

    map_df = pd.DataFrame(mappings, columns=["KEGG_ID", "Pathway"])

    # Filter out global/overview maps (map01100, map01200, etc.) to keep specific functional pathways
    overview_maps = [
        "map01100",
        "map01200",
        "map01210",
        "map01230",
        "map01200",
        "map01110",
        "map01120",
    ]
    map_df = map_df[~map_df["Pathway"].isin(overview_maps)]

    map_df.to_csv(cache_file, sep="\t", index=False)
    print(f"Cached {len(map_df)} KO-pathway links to '{cache_file}'.")
    return map_df


def collapse_kos_to_pathways(input_file, output_file, cache_file):
    """Loads KO abundance table (Rows: KOs, Cols: Samples) and aggregates into Pathways."""
    print(f"Loading KO matrix from {input_file}...")
    # First column is KEGG ID
    df = pd.read_csv(input_file, sep="\t", index_col=0)

    # Clean KEGG ID formatting if needed
    df.index = df.index.astype(str).str.strip()

    # Get Mapping Table
    map_df = get_kegg_mapping_table(cache_file)

    # Merge KO table with pathway map
    merged = map_df.merge(df, left_on="KEGG_ID", right_index=True)

    if merged.empty:
        raise ValueError(
            "No matching KEGG IDs found between input table and KEGG database! "
            "Check if IDs look like 'K00001'."
        )

    print(
        f"Mapped {merged['KEGG_ID'].nunique()} unique KOs into {merged['Pathway'].nunique()} pathways."
    )

    # Group by Pathway and SUM abundances across member KOs per sample
    pathway_df = merged.drop(columns=["KEGG_ID"]).groupby("Pathway").sum()

    # Transpose matrix so rows = Samples, cols = Pathways (ML ready)
    pathway_df_transposed = pathway_df.T

    pathway_df_transposed.to_csv(output_file, sep="\t")
    print(
        f"Pathway matrix saved to '{output_file}' (Shape: {pathway_df_transposed.shape})."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collapse KEGG KO matrix into Pathway matrix."
    )
    parser.add_argument(
        "--input", required=True, help="Path to KO abundance file (TSV)."
    )
    parser.add_argument(
        "--output", required=True, help="Path to save collapsed Pathway matrix."
    )
    parser.add_argument(
        "--cache", default="kegg_ko2pathway.tsv", help="Local KEGG cache file."
    )
    args = parser.parse_args()

    collapse_kos_to_pathways(args.input, args.output, args.cache)

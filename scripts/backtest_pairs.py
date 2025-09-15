import subprocess
import json
import os
import re
import zipfile
import shutil
from pathlib import Path
from freqtrade.configuration.load_config import load_from_files
import pandas as pd
import freqtrade
from pycoingecko import CoinGeckoAPI


# --- Configuration ---
STRATEGY_NAME = "E0V1E"
TIMERANGE = "20250501-20250801"
CONFIG_PATH = Path("configs/volume_pairlists.json")
BACKTEST_RESULTS_DIR = Path("user_data/backtest_results/pairs_analysis/")
CSV_OUTPUT_PATH = Path("./pair_performance.csv")

# --- English Headers for CSV file ---
ENGLISH_HEADERS = {
    "pair": "Trading Pair",
    "coin_price": "Price",
    "coin_market_cap": "Market Cap",
    "coin_volume_24h": "24h Volume",
    "cap_size": "Cap Size",
    "total_profit_pct": "Total Profit (%)",
    "profit_total_abs": "Absolute Profit ($)",
    "market_change_pct": "Market Change (%)",
    "total_trades": "Total Trades",
    "wins": "Wins",
    "losses": "Losses",
    "draws": "Draws",
    "win_rate_pct": "Win Rate (%)",
    "max_drawdown_pct": "Max Drawdown (%)",
    "sharpe_ratio": "Sharpe Ratio",
    "sortino_ratio": "Sortino Ratio",
    "calmar_ratio": "Calmar Ratio",
    "profit_factor": "Profit Factor",
    "expectancy": "Expectancy ($)",
    "cagr": "CAGR (%)",
    "avg_duration_mins": "Avg Hold Time",
    "final_balance": "Final Balance",
}


# Initialize the CoinGecko API client
cg = CoinGeckoAPI()
coin_list = cg.get_coins_list()


def get_coin_raw_data(symbol: str) -> dict | None:
    symbol = symbol.lower()

    coin_ids = [coin["id"] for coin in coin_list if coin.get("symbol") == symbol]
    data = cg.get_price(
        ids=coin_ids,
        vs_currencies="usd",
        include_market_cap="true",
        include_24hr_vol="true",
    )

    data = [{"id": id, **coin} for id, coin in data.items()]

    if not data:
        return None

    return max(data, key=lambda x: x.get("usd_market_cap") or 0)


def format_large_number(value: float) -> str:
    """
    Formats a large number into a human-readable string with currency suffixes.
    Example: 1234567890 -> "$1.23B"
    """
    num = float(value)
    if num >= 1_000_000_000_000:  # Trillions
        return f"${num / 1_000_000_000_000:.2f}T"
    if num >= 1_000_000_000:  # Billions
        return f"${num / 1_000_000_000:.2f}B"
    if num >= 1_000_000:  # Millions
        return f"${num / 1_000_000:.2f}M"
    if num >= 1_000:  # Thousands
        return f"${num / 1_000:.2f}K"
    return f"${num:.2f}"


def get_coin_data(symbol: str) -> dict | None:
    if "/" in symbol:
        symbol = symbol.split("/")[0]

    symbol = symbol.lower()
    coin_data = get_coin_raw_data(symbol)

    if not coin_data:
        return None

    price_raw = coin_data["usd"]
    market_cap_raw = coin_data["usd_market_cap"]
    volume_24h_raw = coin_data["usd_24h_vol"]
    cap_size = get_cap_size_classification(market_cap_raw)

    return {
        "price": price_raw,
        "market_cap": format_large_number(market_cap_raw),
        "volume_24h": format_large_number(volume_24h_raw),
        "cap_size": cap_size,
    }


def get_cap_size_classification(market_cap: float) -> str:
    """
    Classifies a coin's size based on its market capitalization in USD.
    - Large-Cap: > $10 Billion
    - Mid-Cap:   $1 Billion - $10 Billion
    - Small-Cap: $100 Million - $1 Billion
    - Micro-Cap: < $100 Million
    """
    if market_cap is None:
        return "Unknown"

    if market_cap >= 10_000_000_000:
        return "Large-Cap"
    elif market_cap >= 1_000_000_000:
        return "Mid-Cap"
    elif market_cap >= 100_000_000:
        return "Small-Cap"
    else:
        return "Micro-Cap"


def remove_json_comments(json_str: str) -> str:
    """Remove comments from JSON string."""
    return re.sub(r"//.*", "", json_str)


def get_all_pairs() -> list[str]:
    """Get list of all available trading pairs."""
    print("📢 Getting list of all trading pairs...")
    try:
        command = [
            "freqtrade",
            "test-pairlist",
            "--print-json",
            "--config",
            str(CONFIG_PATH),
        ]
        # Hide subprocess output for cleaner interface
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, encoding="utf-8"
        )
        pairs_data = json.loads(result.stdout)
        print(f"✅ Found {len(pairs_data)} active trading pairs.")
        return pairs_data
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        json.JSONDecodeError,
    ) as e:
        print(f"❌ Error getting trading pairs list: {e}")
        return []


def run_backtest_for_pair(pair: str, temp_config_path: Path) -> Path | None:
    """
    Run backtest and find result .zip file through .last_result.json.
    """
    command = [
        "freqtrade",
        "backtesting",
        "--config",
        str(temp_config_path),
        "--strategy",
        STRATEGY_NAME,
        "--timerange",
        TIMERANGE,
        "--export",
        "trades",
        "--export-directory",
        str(BACKTEST_RESULTS_DIR),
        "--max-open-trades",
        "1",
    ]

    try:
        print(f"⏳ Starting backtest for {pair}...")
        # Redirect stdout and stderr to DEVNULL to hide unnecessary logs
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        last_result_info_file = BACKTEST_RESULTS_DIR / ".last_result.json"
        if not last_result_info_file.exists():
            print("❌ Error: .last_result.json file not found.")
            return None

        with open(last_result_info_file, "r") as f:
            data = json.load(f)

        zip_filename = data.get("latest_backtest")
        if not zip_filename:
            print("❌ Error: .last_result.json does not contain 'latest_backtest' key.")
            return None

        latest_file_path = BACKTEST_RESULTS_DIR / zip_filename
        if not latest_file_path.exists():
            print(f"❌ Error: Zip file '{zip_filename}' does not exist.")
            return None

        print(
            f"👍 Backtest successful for {pair}. Result file: {latest_file_path.name}"
        )
        return latest_file_path

    except subprocess.CalledProcessError:
        print(f"⚠️  No data for pair {pair} in the selected time range.")
        return None


def process_and_save_stats(zip_file_path: Path):
    """
    Extract zip file, process JSON result file and update CSV file.
    """
    if not zip_file_path or not zip_file_path.exists():
        return

    temp_extract_dir = BACKTEST_RESULTS_DIR / "temp_extract"
    temp_extract_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(temp_extract_dir)

    json_files_in_zip = list(temp_extract_dir.glob("*.json"))
    # Exclude config file, get main result file
    results_file = next(
        (f for f in json_files_in_zip if not f.name.endswith("_config.json")), None
    )

    if not results_file:
        print(f"❌ Error: No JSON result file found in {zip_file_path.name}")
        return

    with open(results_file, "r") as f:
        stats = json.load(f)

    strategy_stats = stats.get("strategy", {}).get(STRATEGY_NAME, {})
    if not strategy_stats:
        print(f"❌ Error: No data found for strategy '{STRATEGY_NAME}'.")
        return

    pair_specific_stats = next(
        (
            p
            for p in strategy_stats.get("results_per_pair", [])
            if p.get("key") != "TOTAL"
        ),
        {},
    )
    total_trades = pair_specific_stats.get("trades", 0)
    # coin_data = get_coin_data(pair_specific_stats.get("key"))

    pair_data = {
        "pair": pair_specific_stats.get("key"),
        # "coin_price": coin_data.get("price") if coin_data else "no data",
        # "coin_market_cap": coin_data.get("market_cap") if coin_data else "no data",
        # "coin_volume_24h": coin_data.get("volume_24h") if coin_data else "no data",
        # "cap_size": coin_data.get("cap_size") if coin_data else "no data",
        "total_profit_pct": round(pair_specific_stats.get("profit_total_pct", 0.0), 2),
        "profit_total_abs": round(strategy_stats.get("profit_total_abs", 0.0), 3),
        "market_change_pct": round(strategy_stats.get("market_change", 0.0) * 100, 2),
        "total_trades": total_trades,
        "wins": pair_specific_stats.get("wins", 0),
        "losses": pair_specific_stats.get("losses", 0),
        "draws": pair_specific_stats.get("draws", 0),
        "win_rate_pct": (
            round((pair_specific_stats.get("wins", 0) / total_trades) * 100, 2)
            if total_trades > 0
            else 0
        ),
        "max_drawdown_pct": round(
            strategy_stats.get("max_drawdown_account", 0.0) * 100, 2
        ),
        "sharpe_ratio": round(strategy_stats.get("sharpe", 0.0), 3),
        "sortino_ratio": round(strategy_stats.get("sortino", 0.0), 3),
        "calmar_ratio": round(strategy_stats.get("calmar", 0.0), 3),
        "profit_factor": round(strategy_stats.get("profit_factor", 0.0), 3),
        "expectancy": round(strategy_stats.get("expectancy", 0.0), 3),
        "cagr": round(strategy_stats.get("cagr", 0.0) * 100, 2),
        "avg_duration_mins": pair_specific_stats.get("duration_avg", "0:00"),
        "final_balance": round(strategy_stats.get("final_balance", 0.0), 3),
    }

    if pair_data["pair"] is None:
        print("⚠️ Unable to extract pair name. Skipping...")
        return

    df = pd.read_csv(CSV_OUTPUT_PATH) if CSV_OUTPUT_PATH.exists() else pd.DataFrame()

    if not df.empty and pair_data["pair"] in df["Trading Pair"].values:
        df = df[df["Trading Pair"] != pair_data["pair"]]

    new_row_df = pd.DataFrame([pair_data])
    # Rename columns to English before concatenating
    new_row_df.rename(columns=ENGLISH_HEADERS, inplace=True)

    df = pd.concat([df, new_row_df], ignore_index=True)

    if "Total Profit (%)" in df.columns:
        # Sort CSV file by profit descending
        df = df.sort_values(by="Total Profit (%)", ascending=False).reset_index(
            drop=True
        )

    # Reorder columns for better appearance
    ordered_columns = [h for h in ENGLISH_HEADERS.values() if h in df.columns]
    df = df[ordered_columns]

    df.to_csv(CSV_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"💾 Updated results for {pair_data['pair']} in CSV file.")

    if temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir)


def main():
    """Main function to run the entire process."""
    BACKTEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pairs_to_test = get_all_pairs()
    if not pairs_to_test:
        print("No trading pairs found. Exiting.")
        return

    total_pairs = len(pairs_to_test)
    print(f"\n🚀 Starting backtest for {total_pairs} trading pairs...\n")

    temp_config_path = CONFIG_PATH.with_suffix(".tmp.json")
    try:
        with open(CONFIG_PATH, "r") as f:
            config_data = load_from_files([CONFIG_PATH])

        config_data["pairlists"] = [{"method": "StaticPairList"}]
        config_data["max_open_trades"] = 1

        for i, pair in enumerate(pairs_to_test):
            print(f"\n--- 📊 Processing pair {i + 1}/{total_pairs}: {pair} ---")
            config_data["exchange"]["pair_whitelist"] = [pair]

            with open(temp_config_path, "w") as f:
                json.dump(config_data, f, indent=4)

            result_zip_file = run_backtest_for_pair(pair, temp_config_path)

            if result_zip_file:
                process_and_save_stats(result_zip_file)

    finally:
        if temp_config_path.exists():
            os.remove(temp_config_path)
            print("\n🧹 Deleted temporary configuration file.")

    print(f"\n🎉 Completed! Summary results saved at: {CSV_OUTPUT_PATH}")


if __name__ == "__main__":
    main()

import logging
import sys
import os
import pandas as pd
from backtester.data_handler import DataHandler
from backtester.breadth import BreadthCalculator
from backtester.backtester import Backtester
from backtester.kpi_report import KPIReport
from Strategies.sma_crossover import SMACrossover

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

def main():
    try:
        # ----------------------------
        # Data Handling
        # ----------------------------
        folder = r"D:\Algo_trading\Data\Day"
        logging.info("Loading data from folder: %s", folder)
        dh = DataHandler(folder)
        panel = dh.get_panel()

        if panel.empty:
            logging.error("No data found in the folder!")
            return

        logging.info("Data loaded: %d rows x %d columns", panel.shape[0], panel.shape[1])

        # ----------------------------
        # Market Breadth Calculation
        # ----------------------------
        logging.info("Calculating market breadth indicators...")
        bc = BreadthCalculator(dh)
        breadth = bc.compute()

        # ----------------------------
        # Ensure data alignment
        # ----------------------------
        if not panel.index.equals(breadth.index):
            logging.warning("Panel and breadth indexes do not match. Aligning...")
            breadth = breadth.reindex(panel.index).ffill()

        logging.info("Breadth data aligned: %d rows x %d columns", breadth.shape[0], breadth.shape[1])

        # ----------------------------
        # Strategy Setup
        # ----------------------------
        strat = SMACrossover(short=20, long=50)
        logging.info("Generating signals for strategy: %s", strat.name)

        # Check if enough rows exist for SMA calculation
        if len(panel) < strat.long:
            logging.error("Not enough data for SMA calculation. Required: %d rows, available: %d rows",
                          strat.long, len(panel))
            return

        # Generate signals
        signals = strat.generate_signals(panel, breadth)

        if signals.empty:
            logging.warning("No signals generated!")
        else:
            logging.info("Signals generated: %d trades", signals.sum().sum())

        # ----------------------------
        # Backtesting
        # ----------------------------
        logging.info("Running backtest...")
        bt = Backtester(panel)
        eq = bt.run(signals)

        if eq.empty:
            logging.error("Equity curve is empty. Check signals and data.")
            return

        # Save equity and signals
        os.makedirs("D:/Algo_trading/Backtest_Output", exist_ok=True)
        eq.to_csv("D:/Algo_trading/Backtest_Output/equity_curve.csv")
        signals.to_csv("D:/Algo_trading/Backtest_Output/signals.csv")
        logging.info("Equity curve and signals saved to Backtest_Output folder.")

        # ----------------------------
        # KPI Reporting
        # ----------------------------
        report = KPIReport(eq)
        summary = report.summary()
        logging.info("KPI Summary:\n%s", summary)

        # Save KPI report
        pd.DataFrame([summary]).to_csv("D:/Algo_trading/Backtest_Output/kpi_report.csv", index=False)

        # ----------------------------
        # Plotting
        # ----------------------------
        try:
            eq.plot(title=strat.name)
        except Exception as e:
            logging.warning("Could not plot equity curve: %s", e)

        logging.info("Backtest completed successfully.")

    except Exception as e:
        logging.exception("Error running backtest: %s", e)

if __name__ == "__main__":
    main()


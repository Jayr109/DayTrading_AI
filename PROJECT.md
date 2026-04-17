# DayTrading AI — Project Brief

## Goal

Build an ML-based system that connects to live market data, executes futures trades autonomously, and develops its own strategy through paper trading across all market sessions (US, London, Asia).

## Phases (Rough Roadmap)

1. **Data Pipeline** — ingest and store historical OHLCV data for MES and MNQ futures
2. **Model Training** — train ML model on historical data; define reward signal
3. **TradingView Integration** — connect to live market data via TradingView API or broker feed
4. **Paper Trading Engine** — simulate order execution (market, limit, stop-loss, take-profit)
5. **Strategy Discovery** — let model run autonomously across all sessions; log and evaluate strategies
6. **Live Execution** — connect to broker API (e.g., Tradovate, Rithmic) for real order placement

## Instruments
- **MES1** — Micro E-mini S&P 500 futures
- **MNQ1** — Micro E-mini Nasdaq-100 futures

## Sessions to Cover
- US (9:30 AM – 4:00 PM ET)
- London (3:00 AM – 11:30 AM ET)
- Asia (6:00 PM – 3:00 AM ET)

## Key Decisions (To Be Made)
- ML architecture: reinforcement learning (RL) vs. supervised → RL is more natural for strategy discovery
- Data source: TradingView webhook / Tradovate API / Rithmic / Alpaca
- Broker for live execution: Tradovate or Rithmic (both support MES/MNQ)
- Backtesting framework: Backtrader, VectorBT, or custom

## Status
- [ ] Project initialized
- [ ] Data pipeline built
- [ ] First model trained
- [ ] Paper trading running

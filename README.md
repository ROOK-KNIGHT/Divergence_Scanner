# Divergence Scanner

A sophisticated technical analysis tool that scans financial markets for price/indicator divergences using the Schwab API. The scanner identifies bullish and bearish divergences across RSI and Accumulation/Distribution (AD) Volume indicators, providing real-time alerts via Discord webhooks with detailed charts and trade signals.

## Features

### 🔍 Advanced Divergence Detection
- **RSI Divergences**: Detects bullish and bearish divergences between price action and RSI indicator
- **AD Volume Divergences**: Identifies volume-based divergences using Accumulation/Distribution line
- **Classification System**: Categorizes divergences by strength (Strong, Medium, Weak, Hidden)
- **Multi-Timeframe Support**: Configurable timeframes from 1-minute to 4-hour intervals

### 📊 Comprehensive Market Coverage
- **Futures Markets**: /ES, /NQ, /RTY, /YM, /GC, /CL
- **S&P 500 Stocks**: Dynamic fetching of all S&P 500 symbols from Wikipedia
- **Parallel Processing**: Efficient scanning of hundreds of symbols simultaneously
- **Market Hours Detection**: Automatically pauses during market closures

### 🚨 Real-Time Notifications
- **Discord Integration**: Rich embeds with charts, trade signals, and market context
- **Chart Generation**: Automated technical analysis charts with divergence annotations
- **Trade Signals**: Entry/exit levels with risk-reward ratios and stop-loss calculations
- **Cooldown System**: Prevents spam notifications for the same divergences

### 📈 Technical Analysis
- **Multiple Indicators**: RSI, MACD, EMAs, Bollinger Bands, ATR, ADX
- **Swing Detection**: Automated identification of significant price swings
- **Support/Resistance**: Dynamic level detection based on swing clusters
- **Market Sessions**: Asian, European, and US session identification

## Installation

### Prerequisites
- Python 3.8+
- Schwab API credentials
- Discord webhook URL
- TA-Lib library

### Required Python Packages
```bash
pip install pandas numpy talib scipy matplotlib discord-webhook python-dotenv requests
```

### TA-Lib Installation
**Windows:**
```bash
pip install TA-Lib
```

**macOS:**
```bash
brew install ta-lib
pip install TA-Lib
```

**Linux:**
```bash
sudo apt-get install libta-lib-dev
pip install TA-Lib
```

## Configuration

### Environment Variables
Create a `.env` file in the project root with the following variables:

```env
# Schwab API Configuration
SCHWAB_APP_KEY=your_schwab_app_key
SCHWAB_APP_SECRET=your_schwab_app_secret
SCHWAB_REDIRECT_URI=https://127.0.0.1
SCHWAB_TOKEN_FILE=/path/to/your/tokens.json

# Discord Configuration
DISCORD_WEBHOOK_URL=your_discord_webhook_url

# Data Directories
CHARTS_DIR=/path/to/charts/directory

# Optional: AI Model API Keys (for future enhancements)
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
CLAUDE_API_KEY=your_claude_key
```

### Schwab API Setup
1. Register for a Schwab Developer Account
2. Create a new application to get your App Key and Secret
3. Set redirect URI to `https://127.0.0.1`
4. Run the scanner for initial authentication

### Discord Webhook Setup
1. Create a Discord server or use an existing one
2. Go to Server Settings → Integrations → Webhooks
3. Create a new webhook and copy the URL
4. Add the URL to your `.env` file

## Usage

### Basic Usage
```bash
python3 Divergence_calculator.py
```

### Scan Modes
```bash
# Scan futures only
python3 Divergence_calculator.py --mode futures

# Scan stocks only
python3 Divergence_calculator.py --mode stocks

# Scan both (default)
python3 Divergence_calculator.py --mode both
```

### Notification System
```bash
# Run the divergence notifier separately
python3 divergence_notifier.py --live-monitor

# Process saved results
python3 divergence_notifier.py --results-file results.pkl
```

## Project Structure

```
Divergence_Scanner/
├── Divergence_calculator.py    # Main scanner application
├── divergence_notifier.py      # Discord notification system
├── connection_manager.py       # Schwab API authentication
├── historical_data_handler.py  # Market data fetching
├── .env                        # Environment configuration
├── README.md                   # This file
└── divergence_charts/          # Generated chart images
```

## Core Components

### DivergenceScanner Class
The main scanner class that orchestrates the entire process:
- **Parallel Processing**: Scans multiple symbols simultaneously using ThreadPoolExecutor
- **Market Hours**: Automatically detects market open/close times
- **Signal Tracking**: Maintains cooldown periods to prevent duplicate alerts
- **Data Management**: Efficiently handles large datasets with batch processing

### Technical Indicators
- **RSI (Relative Strength Index)**: 14-period RSI for momentum analysis
- **AD Line (Accumulation/Distribution)**: Volume-weighted price indicator
- **MACD**: Moving Average Convergence Divergence for trend confirmation
- **EMAs**: 9, 21, and 50-period exponential moving averages
- **Bollinger Bands**: 20-period bands with 2 standard deviations
- **ATR**: Average True Range for volatility measurement
- **ADX**: Average Directional Index for trend strength

### Divergence Types

#### RSI Divergences
1. **Strong Bullish**: Lower price low + Higher RSI low
2. **Medium Bullish**: Lower price low + Equal RSI low
3. **Weak Bullish**: Lower price low + Lower RSI low (less decline)
4. **Hidden Bullish**: Higher price low + Lower RSI low

5. **Strong Bearish**: Higher price high + Lower RSI high
6. **Medium Bearish**: Higher price high + Equal RSI high
7. **Weak Bearish**: Higher price high + Higher RSI high (less increase)
8. **Hidden Bearish**: Lower price high + Higher RSI high

#### AD Volume Divergences
Similar classification system applied to Accumulation/Distribution line divergences.

### Trade Signal Generation
- **Entry Points**: Based on swing highs/lows where divergences occur
- **Stop Loss**: Calculated using ATR (Average True Range) multiplier
- **Take Profit**: Risk-reward ratio of 2.5:1 minimum
- **Support/Resistance**: Dynamic levels based on swing clusters
- **Market Context**: Considers trend direction and market session

## Discord Notifications

### Rich Embeds
- **Chart Attachments**: Automatically generated technical analysis charts
- **Trade Signals**: Complete entry/exit information with risk metrics
- **Market Context**: Current trend, session, and volatility information
- **Divergence Details**: Timestamps, prices, and indicator values

### Chart Features
- **Candlestick Patterns**: OHLC data with volume
- **Technical Indicators**: RSI, EMAs, and volume overlays
- **Divergence Lines**: Visual connections between swing points
- **Trade Levels**: Entry, stop-loss, and take-profit annotations
- **Support/Resistance**: Dynamic level identification

## Configuration Parameters

### Scanning Parameters
```python
BATCH_SIZE = 200              # Symbols per batch
MAX_WORKERS = 100             # Parallel processing threads
CHECK_INTERVAL = 60           # Scan frequency (seconds)
```

### Technical Parameters
```python
RSI_PERIOD = 14               # RSI calculation period
SWING_LOOKBACK = 10           # Swing detection lookback
MIN_SWING_PERCENT = 0.2       # Minimum swing significance
RISK_REWARD_RATIO = 2.5       # Minimum R:R for trade signals
STOP_LOSS_ATR_MULT = 1.5      # Stop loss ATR multiplier
```

### Market Sessions (UTC)
```python
MARKET_SESSIONS = {
    "Asian": {"start": "21:00", "end": "03:00"},
    "European": {"start": "07:00", "end": "16:00"},
    "US": {"start": "13:30", "end": "20:00"}
}
```

## Performance Optimization

### Parallel Processing
- **ThreadPoolExecutor**: Concurrent symbol scanning
- **Batch Processing**: Efficient API rate limit management
- **Memory Management**: Optimized data structures for large datasets

### API Rate Limiting
- **Exponential Backoff**: Automatic retry with increasing delays
- **Request Throttling**: Controlled API call frequency
- **Error Handling**: Robust exception management

## Logging and Monitoring

### Log Files
- **divergence_scanner.log**: Main application logs
- **divergence_notifier.log**: Notification system logs

### Console Output
- **Real-time Status**: Current scanning progress
- **Summary Statistics**: Divergence counts and signal metrics
- **Error Reporting**: Detailed error messages and stack traces

## Troubleshooting

### Common Issues

1. **TA-Lib Installation**
   ```bash
   # macOS
   brew install ta-lib
   pip install TA-Lib
   
   # Ubuntu/Debian
   sudo apt-get install libta-lib-dev
   pip install TA-Lib
   ```

2. **Schwab API Authentication**
   - Ensure correct App Key and Secret
   - Verify redirect URI matches exactly
   - Check token file permissions

3. **Discord Webhook**
   - Verify webhook URL is correct
   - Check Discord server permissions
   - Test webhook with simple message

### Performance Issues
- Reduce `MAX_WORKERS` if experiencing API rate limits
- Increase `BATCH_SIZE` for faster processing (if API allows)
- Monitor memory usage with large symbol lists

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and informational purposes only. It should not be considered as financial advice. Trading involves substantial risk and may not be suitable for all investors. Past performance does not guarantee future results.

## Support

For issues, questions, or contributions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the configuration parameters

---

**Happy Trading! 📈**

# FUZ CLI (FUCK YOU ZERODHA) by @IndianStreetBets

This project is result of frustration towards Zerodha API. Even for basic API access they are ripping off customers. So, i whipped up a code in an hour or two to get full access to Zerodha API without requiring API Keys. 
The FUZ CLI uses 'Encryption token' method to login to your account. 


![image](https://github.com/user-attachments/assets/665ada07-4267-4550-89b8-643292dcb362)

![image](https://github.com/user-attachments/assets/73dede28-9104-45e1-b874-b9a27073af7d)


# FUZ CLI - WHAT, WHY, WHO.

What : A command-line interface for interacting with the Zerodha trading platform.
Why : Why the fuck not.
Who : Mainly for developers to develop tools on top of it but it can be used by regular end users as well. Functionality has been added .... or atleast i have tried. 

## Project Description

This CLI tool allows users to analyze stocks, manage watchlists, set alerts, and screen stocks based on technical indicators, all from the command line.

## Features

- Stock Analysis
- Market Overview
- Watchlist Management
- Alert System
- Stock Screener
- Account Information

## Current Issues 

- Improve response time
  
## Usage 

This tool is intended for the use of developers and algo traders however end users can still use the features as per usage guide.

FUZ-CLI Usage Guide
        ===================

        1. Analyze a Stock:
           Command: symbol STOCKNAME [TIMEFRAME]
           Example: symbol RELIANCE 5minute
           Available timeframes: 1minute, 3minute, 5minute, 10minute, 15minute, 30minute, 60minute, day

        2. Market Overview:
           Command: market
           This will display the current values of major indices.

        3. Manage Watchlist:
           - Show watchlist: watchlist show
           - Add to watchlist: watchlist add SYMBOL1 SYMBOL2 ...
           - Remove from watchlist: watchlist remove SYMBOL1 SYMBOL2 ...

        4. Manage Alerts:
           - Show alerts: alerts show
           - Add alert: alerts add SYMBOL INDICATOR1 VALUE1 OPERATOR INDICATOR2 VALUE2 TIMEFRAME
             Example: alerts add RELIANCE EMA 7 > EMA 21 5minute
           - Remove alert: alerts remove INDEX

        5. Account Information:
           Command: account

        6. Run Screener:
           Command: screener SECTOR CONDITION1 && CONDITION2 && ... TIMEFRAME
           Example: screener NIFTY50 EMA 7 > EMA 21 && RSI < 30 && VOLUME > SMA 20 15minute

           Available Sectors: NIFTY50, BANK, IT, PHARMA, AUTO, FMCG, METAL, REALTY

           Available Indicators:
           - EMA (Exponential Moving Average): EMA PERIOD
           - SMA (Simple Moving Average): SMA PERIOD
           - RSI (Relative Strength Index): RSI
           - MACD (Moving Average Convergence Divergence): MACD [macd/signal/histogram]
           - BB (Bollinger Bands): BB [upper/middle/lower]
           - ATR (Average True Range): ATR PERIOD
           - OBV (On-Balance Volume): OBV
           - CLOSE (Closing Price): CLOSE
           - VOLUME (Trading Volume): VOLUME

           Condition Format: 
           - For comparisons between indicators: INDICATOR1 VALUE1 OPERATOR INDICATOR2 VALUE2
           - For single indicator conditions: INDICATOR OPERATOR VALUE

           Operators: >, <, ==, >=, <=

           Complex Screener Examples:
           - screener BANK EMA 7 > EMA 21 && RSI < 30 && VOLUME > SMA 20 15minute
           - screener IT MACD > 0 && CLOSE > BB && ATR > 5 5minute
           - screener PHARMA RSI < 30 && CLOSE < BB && OBV > SMA 20 day

        7. Exit:
           Command: exit

        8. Help:
           Command: help

        Note: Replace STOCKNAME, SYMBOL, SECTOR, CONDITIONS with actual values when using the commands.
        Timeframes can be specified as: 1minute, 3minute, 5minute, 10minute, 15minute, 30minute, 60minute, day


## Next steps

- Introduce order placement and order management. 
- Improve current features.
- Remove dependancy of websocket from Zerodha and only use Zerodha websocket for order management.
- Introduce a global scraper to get real-time updates regarding News, Inside buys, Volume spikes, corporate announcements, other events.

## Collaboration

As this is a collaborative effort, algo traders and devs, i request you to provide feedback to improve this further. You can reach out to me on IndianStreetBets Discord server. 

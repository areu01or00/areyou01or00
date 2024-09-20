
#FUZ-CLI by @IndianStreetBets


![image](https://github.com/user-attachments/assets/665ada07-4267-4550-89b8-643292dcb362)

![image](https://github.com/user-attachments/assets/73dede28-9104-45e1-b874-b9a27073af7d)

![image](https://github.com/user-attachments/assets/4fbad485-b10d-485b-b19d-d5fce7a2be04)

![image](https://github.com/user-attachments/assets/6a63f95c-bc9f-448e-977b-751535b3a8d7)



# FUZ CLI - WHAT, WHY, WHO.

- What : A command-line interface for interacting with the Zerodha trading platform.
- Why : Why the fuck not.
- Who : Mainly for developers to develop tools on top of it but it can be used by regular end users as well. Functionality has been added .... or atleast i have tried. 

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
- Improve order placement and management 
  
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

        7. Place Order:
           Command: place_order EXCHANGE SYMBOL TYPE QUANTITY PRODUCT ORDER_TYPE [PRICE]
           Example: place_order NSE RELIANCE BUY 1 MIS MARKET
           Example: place_order NSE INFY SELL 10 CNC LIMIT 1500

        8. Modify Order:
           Command: modify_order ORDER_ID [QUANTITY] [PRICE] [ORDER_TYPE]
           Example: modify_order ORDER123 QUANTITY 5 PRICE 1550

        9. Cancel Order:
           Command: cancel_order ORDER_ID
           Example: cancel_order ORDER123

        10. Get Orders:
            Command: orders

        11. Get Positions:
            Command: positions

        12. Get Holdings:
            Command: holdings

        13. Get Margins:
            Command: margins

        14. Subscribe to Live Data:
            Command: subscribe SYMBOL1 SYMBOL2 ...
            Example: subscribe RELIANCE TCS INFY

        15. Unsubscribe from Live Data:
            Command: unsubscribe SYMBOL1 SYMBOL2 ...
            Example: unsubscribe RELIANCE TCS INFY

        16. Start Live Data Stream:
            Command: start_live_data

        17. Stop Live Data Stream:
            Command: stop_live_data

        18. Exit:
            Command: exit

        19. Help:
            Command: help

        Note: Replace STOCKNAME, SYMBOL, SECTOR, CONDITIONS, EXCHANGE, TYPE, QUANTITY, PRODUCT, ORDER_TYPE, PRICE, and ORDER_ID with actual values when using the commands.
        


## Next steps

- Introduce order placement and order management. 
- Improve current features.
- Remove dependancy of websocket from Zerodha and only use Zerodha websocket for order management.
- Introduce a global scraper to get real-time updates regarding News, Inside buys, Volume spikes, corporate announcements, other events.

## Collaboration

As this is a collaborative effort, algo traders and devs, i request you to provide feedback to improve this further. You can reach out to me on IndianStreetBets Discord server. 

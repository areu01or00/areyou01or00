import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from kite_trade import KiteApp
import json
import shutil
import threading
import time
from colorama import Fore, Back, Style, init
from tqdm import tqdm

init(autoreset=True)

class Indicator:
    @staticmethod
    def SMA(data: pd.Series, period: int) -> pd.Series:
        return data.rolling(window=period).mean()

    @staticmethod
    def EMA(data: pd.Series, period: int) -> pd.Series:
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def RSI(data: pd.Series, period: int) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def MACD(data: pd.Series, fast_period: int, slow_period: int, signal_period: int) -> dict:
        fast_ema = Indicator.EMA(data, fast_period)
        slow_ema = Indicator.EMA(data, slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = Indicator.EMA(macd_line, signal_period)
        return {"macd": macd_line, "signal": signal_line, "histogram": macd_line - signal_line}

class ZerodhaCLI:
    def __init__(self, kite):
        self.kite = kite
        self.watchlist_items = []
        self.cache = {}
        self.indices = [
            "NIFTY 50", "NIFTY BANK", "INDIA VIX", "NIFTY NEXT 50", "NIFTY IT",
            "NIFTY PHARMA", "NIFTY AUTO", "NIFTY FMCG", "NIFTY METAL", "NIFTY REALTY",
            "NIFTY MIDCAP 100", "NIFTY SMALLCAP 100"
        ]
        self.alerts = []
        self.alert_thread = threading.Thread(target=self.check_alerts, daemon=True)
        self.alert_thread.start()
        self.sector_stocks = self.get_sector_stocks()
        self.timeframe_map = {
            "1minute": "minute", "3minute": "3minute", "5minute": "5minute",
            "10minute": "10minute", "15minute": "15minute", "30minute": "30minute",
            "60minute": "60minute", "1hour": "60minute", "day": "day"
        }

    def get_sector_stocks(self):
        return {
            "NIFTY50": [
                "RELIANCE", "TCS", "HDFC", "INFY", "ICICIBANK", "HDFCBANK", "ITC", "KOTAKBANK",
                "HINDUNILVR", "LT", "SBIN", "BHARTIARTL", "BAJFINANCE", "ASIANPAINT", "MARUTI",
                "HCLTECH", "AXISBANK", "WIPRO", "NESTLEIND", "ULTRACEMCO", "SUNPHARMA", "TITAN",
                "TECHM", "BAJAJFINSV", "ONGC", "HDFCLIFE", "NTPC", "POWERGRID", "M&M", "DIVISLAB",
                "JSWSTEEL", "ADANIPORTS", "GRASIM", "BAJAJ-AUTO", "DRREDDY", "TATACONSUM", "COALINDIA",
                "BRITANNIA", "HINDALCO", "TATASTEEL", "SBILIFE", "UPL", "IOC", "EICHERMOT", "CIPLA",
                "TATAMOTORS", "BPCL", "INDUSINDBK", "HEROMOTOCO", "SHREECEM"
            ],
            "BANK": [
                "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN", "INDUSINDBK", "BANDHANBNK",
                "FEDERALBNK", "PNB", "RBLBANK", "IDFCFIRSTB", "BANKBARODA"
            ],
            "IT": [
                "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTTS", "MINDTREE", "MPHASIS", "PERSISTENT",
                "COFORGE", "LTIM", "NAUKRI"
            ],
            "PHARMA": [
                "SUNPHARMA", "DRREDDY", "DIVISLAB", "CIPLA", "BIOCON", "AUROPHARMA", "LUPIN", "TORNTPHARM",
                "ALKEM", "GLAND", "APOLLOHOSP", "SYNGENE"
            ],
            "AUTO": [
                "MARUTI", "M&M", "TATAMOTORS", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO", "TVSMOTOR",
                "BALKRISIND", "MOTHERSON", "BOSCHLTD", "ASHOKLEY", "MRF"
            ],
            "FMCG": [
                "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO", "COLPAL", "GODREJCP",
                "VBL", "TATACONSUM", "UBL", "MCDOWELL-N"
            ],
            "METAL": [
                "TATASTEEL", "HINDALCO", "JSWSTEEL", "ADANIENT", "COAL", "VEDL", "HINDCOPPER", "NATIONALUM",
                "JINDALSTEEL", "APLAPOLLO", "RATNAMANI", "WELCORP"
            ],
            "REALTY": [
                "DLF", "GODREJPROP", "PRESTIGE", "OBEROIRLTY", "PHOENIXLTD", "BRIGADE", "SOBHA",
                "MAHLIFE", "SUNTECK", "IBREALEST"
            ]
        }

    def normalize_timeframe(self, timeframe):
        timeframe = timeframe.lower()
        if timeframe.endswith('s'):
            timeframe = timeframe[:-1]
        return self.timeframe_map.get(timeframe, timeframe)

    def watchlist(self, action, symbols=None):
        if action == "show":
            print(self.create_sub_border("Current Watchlist"))
            print(', '.join(self.watchlist_items))
        elif action == "add":
            for symbol in symbols:
                if symbol not in self.watchlist_items:
                    self.watchlist_items.append(symbol)
            print(self.create_sub_border("Updated Watchlist"))
            print(', '.join(self.watchlist_items))
        elif action == "remove":
            for symbol in symbols:
                if symbol in self.watchlist_items:
                    self.watchlist_items.remove(symbol)
            print(self.create_sub_border("Updated Watchlist"))
            print(', '.join(self.watchlist_items))

    def alerts(self, action, conditions=None):
        if action == "show":
            print(self.create_sub_border("Current Alerts"))
            for i, alert in enumerate(self.alerts):
                print(f"{i+1}. {alert['symbol']}: {alert['condition']} ({alert['timeframe']})")
        elif action == "add":
            if len(conditions) < 6:
                print("Error: Invalid alert format. Use: SYMBOL INDICATOR1 VALUE1 OPERATOR INDICATOR2 VALUE2 TIMEFRAME")
                return
            symbol = conditions[0]
            condition = ' '.join(conditions[1:-1])
            timeframe = self.normalize_timeframe(conditions[-1])
            new_alert = {"symbol": symbol, "condition": condition, "timeframe": timeframe}
            self.alerts.append(new_alert)
            print(self.create_sub_border("Alert Added"))
            print(f"{new_alert['symbol']}: {new_alert['condition']} ({new_alert['timeframe']})")
        elif action == "remove":
            try:
                index = int(conditions[0]) - 1
                removed = self.alerts.pop(index)
                print(self.create_sub_border("Alert Removed"))
                print(f"{removed['symbol']}: {removed['condition']} ({removed['timeframe']})")
            except (IndexError, ValueError):
                print("Error: Invalid alert index.")

    def screener(self, sector, criteria):
        print(self.create_sub_border(f"Stock Screener - {sector.upper()}"))
        try:
            sector = sector.upper()
            if sector in ["BANKNIFTY", "NIFTYBANK"]:
                sector = "BANK"
            
            if sector in self.sector_stocks:
                sector_stocks = self.sector_stocks[sector]
            else:
                print(f"Sector {sector} not found. Available sectors: {', '.join(self.sector_stocks.keys())}")
                return

            parts = criteria.split()
            if len(parts) != 6:
                raise ValueError("Invalid criteria format. Use: INDICATOR1 VALUE1 OPERATOR INDICATOR2 VALUE2 TIMEFRAME")
            
            timeframe = self.normalize_timeframe(parts[-1])
            
            screened_stocks = []
            for stock in tqdm(sector_stocks, desc="Screening Stocks", unit="stock"):
                if self.meets_criteria(stock, ' '.join(parts[:-1] + [timeframe])):
                    screened_stocks.append(stock)

            if screened_stocks:
                print(f"{Fore.GREEN}Stocks meeting the criteria: {', '.join(screened_stocks)}")
            else:
                print(f"{Fore.YELLOW}No stocks met the criteria.")
        except Exception as e:
            self.handle_error(e, "running screener")

    def meets_criteria(self, symbol, criteria):
        try:
            parts = criteria.split()
            timeframe = self.normalize_timeframe(parts[-1])
            
            df = self.get_historical_data(symbol, timeframe, days=30)
            if df is None or df.empty:
                return False
            
            indicator1 = self.calculate_indicator(df, parts[0], int(parts[1]))
            indicator2 = self.calculate_indicator(df, parts[3], int(parts[4]))
            
            return self.evaluate_condition(indicator1.iloc[-1], parts[2], indicator2.iloc[-1])
        except Exception as e:
            self.handle_error(e, f"evaluating criteria for {symbol}")
            return False

    def calculate_indicator(self, df, indicator_name, period):
        if indicator_name == 'EMA':
            return df['close'].ewm(span=period, adjust=False).mean()
        elif indicator_name == 'SMA':
            return df['close'].rolling(window=period).mean()
        else:
            raise ValueError(f"Unknown indicator: {indicator_name}")

    def evaluate_condition(self, value1, operator, value2):
        if operator == '>':
            return value1 > value2
        elif operator == '<':
            return value1 < value2
        elif operator == '==':
            return value1 == value2
        else:
            raise ValueError(f"Unknown operator: {operator}")

    def get_historical_data(self, symbol, timeframe, days=30):
        try:
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            instruments = self.kite.instruments("NSE")
            instrument = next((i for i in instruments if i['tradingsymbol'] == symbol), None)
            
            if instrument is None:
                raise ValueError(f"Symbol {symbol} not found in NSE instruments")
            
            data = self.kite.historical_data(instrument['instrument_token'], from_date, to_date, timeframe)
            
            if data:
                df = pd.DataFrame(data)
                df.set_index('date', inplace=True)
                return df
            else:
                print(f"No historical data available for {symbol}")
                return None
        except Exception as e:
            self.handle_error(e, f"fetching historical data for {symbol}")
            return None

    def stock_analysis(self, symbol, indicators, timeframe="day"):
        try:
            timeframe = self.normalize_timeframe(timeframe)
            print(self.create_sub_border(f"Fetching data for {symbol} on {timeframe} timeframe..."))
            
            df = self.get_historical_data(symbol, timeframe)
            if df is None or df.empty:
                print(f"Error: No historical data available for {symbol} on {timeframe} timeframe.")
                return

            analysis = {
                'symbol': symbol,
                'current_price': df['close'].iloc[-1],
                '30d_high': df['high'].max(),
                '30d_low': df['low'].min(),
                '30d_avg_volume': df['volume'].mean(),
                '30d_price_change': ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100)
            }

            print(self.create_sub_border("Stock Analysis"))
            for key, value in analysis.items():
                print(f"{key}: {value}")

            print(self.create_sub_border("Recent Data"))
            print(df.tail().to_string())

            self.plot_ascii_chart(df)

            print(self.create_sub_border("Calculating indicators..."))
            for indicator in indicators:
                try:
                    if indicator.startswith('SMA') or indicator.startswith('EMA'):
                        period = int(indicator.split('-')[1])
                        df[indicator] = getattr(Indicator, indicator.split('-')[0])(df['close'], period)
                    elif indicator == 'RSI':
                        df['RSI'] = Indicator.RSI(df['close'], 14)
                    elif indicator == 'MACD':
                        macd_data = Indicator.MACD(df['close'], 12, 26, 9)
                        df['MACD'] = macd_data['macd']
                        df['Signal'] = macd_data['signal']
                        df['MACD Histogram'] = macd_data['histogram']
                    print(f"{indicator} calculated successfully.")
                except Exception as e:
                    self.handle_error(e, f"calculating {indicator}")

            print(self.create_sub_border("Indicator Values (latest)"))
            for col in df.columns:
                if col not in ['open', 'high', 'low', 'close', 'volume']:
                    print(f"{col}: {df[col].iloc[-1]:.2f}")

        except Exception as e:
            self.handle_error(e, f"analyzing stock {symbol}")

    def plot_ascii_chart(self, data, width=80, height=20):
        """Plot an ASCII candlestick chart of the given data."""
        if len(data) == 0:
            return
        
        # Use the last 'width' number of data points
        data = data.iloc[-width:]
        
        opens = data['open']
        highs = data['high']
        lows = data['low']
        closes = data['close']
        
        min_val, max_val = lows.min(), highs.max()
        price_range = max_val - min_val
        
        def normalize(val):
            return int((val - min_val) / price_range * (height - 1))
        
        chart = [[' ' for _ in range(width)] for _ in range(height)]
        
        for i, (o, h, l, c) in enumerate(zip(opens, highs, lows, closes)):
            no, nh, nl, nc = map(normalize, (o, h, l, c))
            
            # Draw the wick
            for j in range(nl, nh + 1):
                chart[height - 1 - j][i] = '│'
            
            # Draw the body
            body_start, body_end = min(no, nc), max(no, nc)
            body_char = '█' if c >= o else '░'
            for j in range(body_start, body_end + 1):
                chart[height - 1 - j][i] = body_char
        
        # Add y-axis labels
        for i in range(height):
            price = max_val - (i / (height - 1)) * price_range
            chart[i] = [f"{price:7.2f} |"] + chart[i]
        
        # Print the chart
        print(self.create_sub_border("Candlestick Chart"))
        print('\n'.join(''.join(row) for row in chart))
        
        # Print x-axis
        dates = data.index
        date_step = max(width // 10, 1)
        x_axis = "        " + " " * 7 + "".join([dates[i].strftime('%Y-%m-%d').center(date_step) for i in range(0, width, date_step)])
        print(x_axis)
        
        # Print summary
        print(f"\n{Fore.YELLOW}Open: {opens.iloc[-1]:.2f} {Fore.GREEN}High: {highs.iloc[-1]:.2f} ", end="")
        print(f"{Fore.RED}Low: {lows.iloc[-1]:.2f} {Fore.CYAN}Close: {closes.iloc[-1]:.2f} ", end="")
        print(f"{Fore.MAGENTA}Volume: {data['volume'].iloc[-1]:,}")

    def market_overview(self):
        try:
            print(self.create_sub_border("Market Overview"))
            for index in self.indices:
                instrument = self.kite.instruments("NSE")
                index_instrument = next((i for i in instrument if i['tradingsymbol'] == index), None)
                if index_instrument:
                    to_date = datetime.now()
                    from_date = to_date - timedelta(minutes=5)  # Get last 5 minutes of data
                    data = self.kite.historical_data(index_instrument['instrument_token'], from_date, to_date, "minute")
                    if data:
                        last_price = data[-1]['close']
                        prev_close = data[0]['open']
                        change = (last_price - prev_close) / prev_close * 100
                        color = Fore.GREEN if change >= 0 else Fore.RED
                        print(f"{index:<15}: {color}{last_price:.2f} (Change: {change:.2f}%)")
                    else:
                        print(f"{index:<15}: {Fore.YELLOW}Data not available")
                else:
                    print(f"{index:<15}: {Fore.YELLOW}Index not found")
            print(self.create_sub_border(""))
        except Exception as e:
            self.handle_error(e, "fetching market overview")

    def account_info(self):
        try:
            profile = self.kite.profile()
            margins = self.kite.margins()
            positions = self.kite.positions()

            print(self.create_sub_border("Account Information"))
            print(f"{Fore.CYAN}Name: {profile['user_name']}")
            print(f"{Fore.CYAN}Email: {profile['email']}")
            print(f"{Fore.GREEN}Available Cash: ₹{margins['equity']['available']['cash']:.2f}")
            print(f"{Fore.YELLOW}Used Margin: ₹{margins['equity']['utilised']['debits']:.2f}")

            print(self.create_sub_border("Current Positions"))
            if positions['net']:
                for position in positions['net']:
                    print(f"{Fore.CYAN}{position['tradingsymbol']}: {position['quantity']} @ ₹{position['average_price']:.2f}")
            else:
                print(f"{Fore.YELLOW}No open positions.")
        except Exception as e:
            self.handle_error(e, "fetching account information")

    def check_alerts(self):
        while True:
            for alert in self.alerts:
                try:
                    symbol = alert['symbol']
                    condition = alert['condition']
                    timeframe = alert['timeframe']
                    
                    df = self.get_historical_data(symbol, timeframe, days=1)
                    if df is None or df.empty:
                        continue
                    
                    parts = condition.split()
                    indicator1 = self.calculate_indicator(df, parts[0], int(parts[1]))
                    indicator2 = self.calculate_indicator(df, parts[3], int(parts[4]))
                    
                    if self.evaluate_condition(indicator1.iloc[-1], parts[2], indicator2.iloc[-1]):
                        print(f"\nAlert triggered: {symbol} {condition} ({timeframe})")
                
                except Exception as e:
                    self.handle_error(e, f"checking alert for {symbol}")
            
            time.sleep(60)  # Check every minute

    def handle_error(self, error, context):
        error_message = f"An error occurred while {context}: {str(error)}"
        suggestion = ""
        
        if "NoneType" in str(error):
            suggestion = "This might be due to missing data. Please check your internet connection and try again."
        elif "out-of-bounds" in str(error):
            suggestion = "This could be caused by insufficient data. Try a different time range or symbol."
        elif "invalid literal for int()" in str(error):
            suggestion = "Please ensure you're entering numeric values where required."
        
        if suggestion:
            error_message += f"\nSuggestion: {suggestion}"
        
        print(self.create_sub_border("Error"))
        print(Fore.RED + error_message)

    def create_sub_border(self, title):
        term_width, _ = shutil.get_terminal_size()
        return f"\n{'-' * term_width}\n{Fore.CYAN}{title}{Fore.RESET}\n{'-' * term_width}"

    def print_usage_guide(self):
        guide = """
        Zerodha CLI Tool Usage Guide
        ============================

        1. Analyze a Stock:
           Command: symbol STOCKNAME [TIMEFRAME]
           Example: symbol RELIANCE 5minute

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
             Example: alerts add RELIANCE 7 EMA > 21 EMA 5minute
           - Remove alert: alerts remove INDEX

        5. Account Information:
           Command: account

        6. Run Screener:
           Command: screener SECTOR INDICATOR1 VALUE1 OPERATOR INDICATOR2 VALUE2 TIMEFRAME
           Example: screener NIFTY50 7 EMA > 21 EMA 5minute

        7. Exit:
           Command: exit

        8. Help:
           Command: help

        Available Sectors for Screener:
        NIFTY50, BANK, IT, PHARMA, AUTO, FMCG, METAL, REALTY

        Note: Replace STOCKNAME, SYMBOL, SECTOR, CONDITIONS with actual values when using the commands.
        """
        print(self.create_sub_border("Zerodha CLI Help"))
        print(guide)

def main():
    enctoken = input("Enter your Zerodha encryption token: ")
    kite = KiteApp(enctoken=enctoken)
    cli = ZerodhaCLI(kite)

    print(cli.create_sub_border("Successfully connected to Zerodha!"))

    while True:
        try:
            user_input = input(f"{Fore.GREEN}Zerodha CLI> {Fore.RESET}").split()
            command = user_input[0] if user_input else ""

            if command == "symbol":
                if len(user_input) < 2:
                    print("Usage: symbol STOCKNAME [TIMEFRAME]")
                elif len(user_input) == 2:
                    cli.stock_analysis(user_input[1], ["SMA-50", "EMA-20", "RSI", "MACD"])
                else:
                    cli.stock_analysis(user_input[1], ["SMA-50", "EMA-20", "RSI", "MACD"], user_input[2])
            elif command == "market":
                cli.market_overview()
            elif command == "watchlist":
                if len(user_input) < 2:
                    print("Usage: watchlist [show|add|remove] [SYMBOLS...]")
                elif user_input[1] == "show":
                    cli.watchlist("show")
                elif user_input[1] in ["add", "remove"] and len(user_input) > 2:
                    cli.watchlist(user_input[1], user_input[2:])
                else:
                    print("Invalid watchlist command")
            elif command == "alerts":
                if len(user_input) < 2:
                    print("Usage: alerts [show|add|remove] [CONDITIONS...]")
                elif user_input[1] == "show":
                    cli.alerts("show")
                elif user_input[1] in ["add", "remove"] and len(user_input) > 2:
                    cli.alerts(user_input[1], user_input[2:])
                else:
                    print("Invalid alerts command")
            elif command == "account":
                cli.account_info()
            elif command == "screener":
                if len(user_input) < 8:
                    print("Usage: screener SECTOR INDICATOR1 VALUE1 OPERATOR INDICATOR2 VALUE2 TIMEFRAME")
                else:
                    cli.screener(user_input[1], ' '.join(user_input[2:]))
            elif command == "exit":
                print(cli.create_sub_border("Thank you for using Zerodha CLI. Goodbye!"))
                break
            elif command == "help":
                cli.print_usage_guide()
            else:
                print("Invalid command. Type 'help' for a list of commands.")
        except Exception as e:
            cli.handle_error(e, "processing command")

if __name__ == "__main__":
    main()
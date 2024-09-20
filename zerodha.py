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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize colorama for colored console output
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

    @staticmethod
    def Bollinger_Bands(data: pd.Series, period: int, std_dev: int) -> dict:
        sma = Indicator.SMA(data, period)
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return {"upper": upper_band, "middle": sma, "lower": lower_band}

    @staticmethod
    def ATR(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def OBV(close: pd.Series, volume: pd.Series) -> pd.Series:
        return (np.sign(close.diff()) * volume).cumsum()

class EnhancedCache:
    def __init__(self, expiry=300):
        self.cache = {}
        self.expiry = expiry

    def get(self, key):
        if key in self.cache:
            if time.time() - self.cache[key]['timestamp'] < self.expiry:
                return self.cache[key]['data']
        return None

    def set(self, key, value):
        self.cache[key] = {
            'data': value,
            'timestamp': time.time()
        }

class FUZCLI:
    def __init__(self, kite):
        self.kite = kite
        self.watchlist_items = []
        self.cache = EnhancedCache()
        self.indices = {
            "NIFTY 50": "NIFTY 50",
            "NIFTY BANK": "NIFTY BANK",
            "INDIA VIX": "INDIA VIX",
            "NIFTY NEXT 50": "NIFTY NEXT 50",
            "NIFTY IT": "NIFTY IT",
            "NIFTY PHARMA": "NIFTY PHARMA",
            "NIFTY AUTO": "NIFTY AUTO",
            "NIFTY FMCG": "NIFTY FMCG",
            "NIFTY METAL": "NIFTY METAL",
            "NIFTY REALTY": "NIFTY REALTY",
            "NIFTY MIDCAP 100": "NIFTY MIDCAP 100",
            "NIFTY SMLCAP 100": "NIFTY SMLCAP 100"
        }
        self.alerts = []
        self.alert_thread = threading.Thread(target=self.check_alerts, daemon=True)
        self.alert_thread.start()
        self.sector_stocks = self.get_sector_stocks()
        self.timeframe_map = {
            "1minute": "minute", "3minute": "3minute", "5minute": "5minute",
            "10minute": "10minute", "15minute": "15minute", "30minute": "30minute",
            "60minute": "60minute", "1hour": "60minute", "day": "day"
        }
        self.instruments = self.get_instruments()

    def get_instruments(self):
        instruments = self.cache.get('instruments')
        if not instruments:
            instruments = {i['tradingsymbol']: i for i in self.kite.instruments("NSE")}
            self.cache.set('instruments', instruments)
        return instruments

    def get_sector_stocks(self):
        return {
            "NIFTY50": [
                "RELIANCE", "TCS", "INFY", "ICICIBANK", "HDFCBANK", "ITC", "KOTAKBANK",
                "HINDUNILVR", "LT", "SBIN", "BHARTIARTL", "BAJFINANCE", "ASIANPAINT", "MARUTI",
                "HCLTECH", "AXISBANK", "WIPRO", "NESTLEIND", "ULTRACEMCO", "SUNPHARMA", "TITAN",
                "TECHM", "BAJAJFINSV", "ONGC", "HDFCLIFE", "NTPC", "POWERGRID", "M&M", "DIVISLAB",
                "JSWSTEEL", "ADANIPORTS", "GRASIM", "BAJAJ-AUTO", "DRREDDY", "TATACONSUM", "COALINDIA",
                "BRITANNIA", "HINDALCO", "TATASTEEL", "SBILIFE", "UPL", "IOC", "EICHERMOT", "CIPLA",
                "TATAMOTORS", "BPCL", "INDUSINDBK", "HEROMOTOCO", "SHREECEM"
            ],
            "BANK": [
                "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN", "INDUSINDBK", "BANDHANBNK",
                "FEDERALBNK", "IDFCFIRSTB", "PNB", "RBLBANK", "BANKBARODA", "AUBANK", "BANKINDIA"
            ],
            "IT": [
                "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTTS", "MINDTREE", "MPHASIS", "COFORGE",
                "PERSISTENT", "OFSS", "LTIM", "NAUKRI"
            ],
            "PHARMA": [
                "SUNPHARMA", "DRREDDY", "DIVISLAB", "CIPLA", "BIOCON", "AUROPHARMA", "LUPIN", "ALKEM",
                "TORNTPHARM", "APOLLOHOSP", "GLAND", "IPCALAB", "NATCOPHARM", "SYNGENE"
            ],
            "AUTO": [
                "MARUTI", "M&M", "TATAMOTORS", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO", "BALKRISIND",
                "BOSCHLTD", "MOTHERSON", "TVSMOTOR", "ASHOKLEY", "BHARATFORG", "EXIDEIND", "SONACOMS"
            ],
            "FMCG": [
                "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO", "COLPAL", "GODREJCP",
                "VBL", "TATACONSUM", "UBL", "MCDOWELL-N", "EMAMILTD", "PGHH"
            ],
            "METAL": [
                "TATASTEEL", "HINDALCO", "JSWSTEEL", "ADANIENT", "COAL", "VEDL", "JINDALSTEL",
                "APLAPOLLO", "HINDZINC", "NMDC", "NATIONALUM", "RATNAMANI", "MOIL", "SAIL"
            ],
            "REALTY": [
                "DLF", "GODREJPROP", "PRESTIGE", "OBEROIRLTY", "PHOENIXLTD", "SUNTECK", "BRIGADE",
                "MAHLIFE", "SOBHA", "IBREALEST", "LODHA"
            ]
        }

    def normalize_timeframe(self, timeframe):
        timeframe = timeframe.lower()
        if timeframe.endswith('s'):
            timeframe = timeframe[:-1]
        normalized = self.timeframe_map.get(timeframe, timeframe)
        if normalized not in self.timeframe_map.values():
            raise ValueError(f"Invalid timeframe: {timeframe}")
        return normalized

    def get_historical_data_batch(self, symbols, timeframe, days=30):
        try:
            normalized_timeframe = self.normalize_timeframe(timeframe)
        except ValueError as e:
            print(f"Error: {str(e)}")
            return {}

        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        def fetch_data(symbol):
            try:
                cache_key = f"{symbol}:{timeframe}:{from_date}:{to_date}"
                cached_data = self.cache.get(cache_key)
                if cached_data is not None:
                    return symbol, cached_data

                instrument = self.instruments.get(symbol)
                if instrument is None:
                    print(f"Symbol {symbol} not found in instruments")
                    return symbol, None
                instrument_token = instrument['instrument_token']
                data = self.kite.historical_data(instrument_token, from_date, to_date, normalized_timeframe)
                if not data:
                    print(f"No data available for {symbol}")
                    return symbol, None
                df = pd.DataFrame(data).set_index('date')
                self.cache.set(cache_key, df)
                return symbol, df
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
                return symbol, None

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(fetch_data, symbols))

        return {symbol: df for symbol, df in results if df is not None}

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

            conditions = [cond.strip() for cond in criteria.split('&&')]
            timeframe = self.normalize_timeframe(conditions[-1].split()[-1])
            conditions = conditions[:-1]  # Remove timeframe from conditions
            
            stock_data = self.get_historical_data_batch(sector_stocks, timeframe, days=30)
            
            screened_stocks = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_stock = {executor.submit(self.meets_multiple_criteria, stock, stock_data.get(stock), conditions): stock for stock in stock_data}
                for future in tqdm(as_completed(future_to_stock), total=len(future_to_stock), desc="Screening Stocks"):
                    stock = future_to_stock[future]
                    try:
                        if future.result():
                            screened_stocks.append(stock)
                    except Exception as exc:
                        print(f'{stock} generated an exception: {exc}')

            if screened_stocks:
                print(f"{Fore.GREEN}Stocks meeting the criteria: {', '.join(screened_stocks)}")
                self.plot_screened_stocks(screened_stocks, stock_data)
            else:
                print(f"{Fore.YELLOW}No stocks met the criteria.")
        except Exception as e:
            self.handle_error(e, "running screener")

    def meets_multiple_criteria(self, symbol, df, conditions):
        try:
            if df is None:
                return False
            for condition in conditions:
                if not self.meets_criteria(df, condition):
                    return False
            return True
        except Exception as e:
            self.handle_error(e, f"evaluating criteria for {symbol}")
            return False

    def meets_criteria(self, df, criteria):
        try:
            parts = criteria.split()
            if len(parts) == 3:  # For conditions like "RSI < 30"
                indicator = self.calculate_indicator(df, parts[0])
                operator = parts[1]
                value = float(parts[2])
                return self.evaluate_condition(indicator.iloc[-1], operator, value)
            elif len(parts) == 5:  # For conditions like "EMA 7 > EMA 21"
                indicator1 = self.calculate_indicator(df, parts[0], parts[1])
                operator = parts[2]
                indicator2 = self.calculate_indicator(df, parts[3], parts[4])
                return self.evaluate_condition(indicator1.iloc[-1], operator, indicator2.iloc[-1])
            else:
                raise ValueError(f"Invalid criteria format: {criteria}")
        except Exception as e:
            self.handle_error(e, f"evaluating criteria: {criteria}")
            return False

    def calculate_indicator(self, df, indicator_name, value=None):
        if indicator_name == 'EMA':
            return Indicator.EMA(df['close'], int(value))
        elif indicator_name == 'SMA':
            return Indicator.SMA(df['close'], int(value))
        elif indicator_name == 'RSI':
            return Indicator.RSI(df['close'], 14)  # Using default period of 14 for RSI
        elif indicator_name == 'MACD':
            macd_data = Indicator.MACD(df['close'], 12, 26, 9)
            return macd_data['macd'] if value is None else macd_data[value.lower()]
        elif indicator_name == 'BB':
            bb_data = Indicator.Bollinger_Bands(df['close'], 20, 2)
            return bb_data['middle'] if value is None else bb_data[value.lower()]
        elif indicator_name == 'ATR':
            return Indicator.ATR(df['high'], df['low'], df['close'], 14 if value is None else int(value))
        elif indicator_name == 'OBV':
            return Indicator.OBV(df['close'], df['volume'])
        elif indicator_name == 'CLOSE':
            return df['close']
        elif indicator_name == 'VOLUME':
            return df['volume']
        else:
            try:
                return float(indicator_name)
            except ValueError:
                raise ValueError(f"Unknown indicator: {indicator_name}")

    def evaluate_condition(self, value1, operator, value2):
        if operator == '>':
            return value1 > value2
        elif operator == '<':
            return value1 < value2
        elif operator == '==':
            return value1 == value2
        elif operator == '>=':
            return value1 >= value2
        elif operator == '<=':
            return value1 <= value2
        else:
            raise ValueError(f"Unknown operator: {operator}")

    def plot_screened_stocks(self, stocks, stock_data):
        term_width, _ = shutil.get_terminal_size()
        chart_width = 40
        charts_per_row = max(1, term_width // chart_width)
        
        for i in range(0, len(stocks), charts_per_row):
            stock_subset = stocks[i:i+charts_per_row]
            charts = []
            for stock in stock_subset:
                if stock in stock_data:
                    charts.append(self.create_mini_chart(stock_data[stock], stock, width=chart_width-10, height=10))
                else:
                    charts.append(f"No data for {stock}")
            
            max_lines = max(len(chart.split('\n')) for chart in charts)
            for line in range(max_lines):
                for chart in charts:
                    lines = chart.split('\n')
                    if line < len(lines):
                        print(lines[line].ljust(chart_width), end='')
                    else:
                        print(' ' * chart_width, end='')
                print()
            print()

    def create_mini_chart(self, data, symbol, width=30, height=10):
        if len(data) == 0:
            return f"No data for {symbol}"
        
        opens = data['open']
        highs = data['high']
        lows = data['low']
        closes = data['close']
        volumes = data['volume']
        
        min_val, max_val = lows.min(), highs.max()
        price_range = max_val - min_val
        
        def normalize(val):
            return int((val - min_val) / price_range * (height - 1))
        
        chart = [[' ' for _ in range(width)] for _ in range(height)]
        
        for i, (o, h, l, c) in enumerate(zip(opens.iloc[-width:], highs.iloc[-width:], lows.iloc[-width:], closes.iloc[-width:])):
            no, nh, nl, nc = map(normalize, (o, h, l, c))
            
            for j in range(nl, nh + 1):
                chart[height - 1 - j][i] = '│'
            
            body_start, body_end = min(no, nc), max(no, nc)
            body_char = '█' if c >= o else '░'
            for j in range(body_start, body_end + 1):
                chart[height - 1 - j][i] = body_char
        
        last_close = closes.iloc[-1]
        change = (last_close - opens.iloc[-1]) / opens.iloc[-1] * 100
        volume = volumes.iloc[-1]
        color = Fore.GREEN if change >= 0 else Fore.RED
        
        info_box = [
            f"┌{'─' * (width + 2)}┐",
            f"│ {symbol:<{width}} │",
            f"│ Close: {color}{last_close:.2f} ({change:+.2f}%){Fore.RESET} │",
            f"│ Volume: {volume:,} │",
            f"└{'─' * (width + 2)}┘"
        ]
        
        chart_str = '\n'.join(info_box + [''.join(row) for row in chart])
        return chart_str

    def market_overview(self):
        try:
            print(self.create_sub_border("Market Overview"))
            index_data = self.get_historical_data_batch(list(self.indices.values()), "minute", days=1)
            
            for display_name, index in self.indices.items():
                if index in index_data:
                    df = index_data[index]
                    if not df.empty:
                        last_price = df['close'].iloc[-1]
                        prev_close = df['open'].iloc[0]
                        change = (last_price - prev_close) / prev_close * 100
                        color = Fore.GREEN if change >= 0 else Fore.RED
                        print(f"{display_name:<15}: {color}{last_price:.2f} (Change: {change:.2f}%)")
                    else:
                        print(f"{display_name:<15}: {Fore.YELLOW}Data not available")
                else:
                    print(f"{display_name:<15}: {Fore.YELLOW}Index not found")
            print(self.create_sub_border(""))
        except Exception as e:
            self.handle_error(e, "fetching market overview")

    def stock_analysis(self, symbol, indicators, timeframe="day"):
        try:
            timeframe = self.normalize_timeframe(timeframe)
            print(self.create_sub_border(f"Fetching data for {symbol} on {timeframe} timeframe..."))
            
            stock_data = self.get_historical_data_batch([symbol], timeframe, days=30)
            if symbol not in stock_data or stock_data[symbol] is None or stock_data[symbol].empty:
                print(f"Error: No historical data available for {symbol} on {timeframe} timeframe.")
                return

            df = stock_data[symbol]

            analysis = {
                'symbol': symbol,
                'current_price': self.safe_float(df['close'].iloc[-1]),
                '30d_high': self.safe_float(df['high'].max()),
                '30d_low': self.safe_float(df['low'].min()),
                '30d_avg_volume': self.safe_float(df['volume'].mean()),
                '30d_price_change': self.safe_float(((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100))
            }

            print(self.create_sub_border("Stock Analysis"))
            for key, value in analysis.items():
                if key in ['current_price', '30d_price_change']:
                    color = Fore.GREEN if value >= 0 else Fore.RED
                    print(f"{key}: {color}{value:.2f}" if isinstance(value, float) else f"{key}: {color}{value}")
                else:
                    print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")

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
                    value = df[col].iloc[-1]
                    color = Fore.GREEN if self.safe_float(value) >= 0 else Fore.RED
                    print(f"{col}: {color}{value:.2f}" if isinstance(value, float) else f"{col}: {color}{value}")

        except Exception as e:
            self.handle_error(e, f"analyzing stock {symbol}")

    def safe_float(self, value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def plot_ascii_chart(self, data, width=80, height=20):
        if len(data) == 0:
            return
        
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
            
            for j in range(nl, nh + 1):
                chart[height - 1 - j][i] = '│'
            
            body_start, body_end = min(no, nc), max(no, nc)
            body_char = '█' if c >= o else '░'
            for j in range(body_start, body_end + 1):
                chart[height - 1 - j][i] = body_char
        
        for i in range(height):
            price = max_val - (i / (height - 1)) * price_range
            chart[i] = [f"{price:7.2f} |"] + chart[i]
        
        print(self.create_sub_border("Candlestick Chart"))
        print('\n'.join(''.join(row) for row in chart))
        
        dates = data.index
        date_step = max(width // 10, 1)
        x_axis = "        " + " " * 7 + "".join([dates[i].strftime('%Y-%m-%d').center(date_step) for i in range(0, width, date_step)])
        print(x_axis)
        
        open_color = Fore.GREEN if closes.iloc[-1] >= opens.iloc[-1] else Fore.RED
        close_color = Fore.GREEN if closes.iloc[-1] >= opens.iloc[-1] else Fore.RED
        print(f"\n{open_color}Open: {opens.iloc[-1]:.2f} {Fore.GREEN}High: {highs.iloc[-1]:.2f} ", end="")
        print(f"{Fore.RED}Low: {lows.iloc[-1]:.2f} {close_color}Close: {closes.iloc[-1]:.2f} ", end="")
        print(f"{Fore.MAGENTA}Volume: {data['volume'].iloc[-1]:,}")

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
                    pnl = position['pnl']
                    color = Fore.GREEN if pnl >= 0 else Fore.RED
                    print(f"{Fore.CYAN}{position['tradingsymbol']}: {position['quantity']} @ ₹{position['average_price']:.2f} | PNL: {color}₹{pnl:.2f}")
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
                    
                    df = self.get_historical_data_batch([symbol], timeframe, days=1)[symbol]
                    if df is None or df.empty:
                        continue
                    
                    if self.meets_criteria(df, condition):
                        print(f"\nAlert triggered: {symbol} {condition} ({timeframe})")
                
                except Exception as e:
                    self.handle_error(e, f"checking alert for {symbol}")
            
            time.sleep(60)  # Check every minute

    def handle_error(self, error, context):
        error_message = f"An error occurred while {context}: {str(error)}"
        print(self.create_sub_border("Error"))
        print(Fore.RED + error_message)

    def create_sub_border(self, title):
        term_width, _ = shutil.get_terminal_size()
        return f"\n{'-' * term_width}\n{Fore.CYAN}{title}{Fore.RESET}\n{'-' * term_width}"

    def print_usage_guide(self):
        guide = """
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
        """
        print(self.create_sub_border("FUZ-CLI Help"))
        print(guide)

def main():
    print(f"{Fore.CYAN}Welcome to FUZ-CLI - Your Zerodha Trading Assistant!{Fore.RESET}")
    print(f"{Fore.YELLOW}Please input your enctoken{Fore.RESET}")
    
    enctoken = input("Enter your Zerodha encryption token: ")
    
    try:
        kite = KiteApp(enctoken=enctoken)
        cli = FUZCLI(kite)
        print(cli.create_sub_border("Successfully connected to FUZ-CLI!"))
        print(f"{Fore.GREEN}Type 'help' for a list of available commands.{Fore.RESET}")

        while True:
            try:
                user_input = input(f"{Fore.GREEN}FUZ-CLI> {Fore.RESET}").split()
                command = user_input[0].lower() if user_input else ""

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
                    if len(user_input) < 3:
                        print("Usage: screener SECTOR CONDITION1 && CONDITION2 && ... TIMEFRAME")
                    else:
                        sector = user_input[1]
                        criteria = ' '.join(user_input[2:])
                        cli.screener(sector, criteria)
                elif command == "exit":
                    print(cli.create_sub_border("Thank you for using FUZ-CLI. Goodbye!"))
                    break
                elif command == "help":
                    cli.print_usage_guide()
                else:
                    print("Invalid command. Type 'help' for a list of commands.")
            except Exception as e:
                cli.handle_error(e, "processing command")

    except Exception as e:
        print(f"{Fore.RED}Error initializing FUZ-CLI: {str(e)}{Fore.RESET}")
        print(f"{Fore.YELLOW}Please check your encryption token and try again.{Fore.RESET}")

if __name__ == "__main__":
    main()

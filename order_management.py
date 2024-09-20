from kite_trade import KiteApp
import logging
import time
import threading
import random
from datetime import datetime, time as dt_time
import pytz

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OrderManager:
    def __init__(self, kite: KiteApp):
        self.kite = kite

    def place_order(self, exchange, symbol, transaction_type, quantity, product, order_type, price=None):
        try:
            order = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=exchange,
                tradingsymbol=symbol,
                transaction_type=transaction_type,
                quantity=quantity,
                product=product,
                order_type=order_type,
                price=price
            )
            if order is None:
                raise ValueError("Order placement failed. This might be due to market closure or API issues.")
            logger.info(f"Order placed successfully. Order ID: {order['order_id']}")
            return order['order_id']
        except Exception as e:
            if isinstance(e, ValueError):
                logger.error(str(e))
            else:
                logger.error(f"Error placing order: {str(e)}")
            raise

    def modify_order(self, order_id, quantity=None, price=None, order_type=None):
        try:
            self.kite.modify_order(
                variety=self.kite.VARIETY_REGULAR,
                order_id=order_id,
                quantity=quantity,
                price=price,
                order_type=order_type
            )
            logger.info(f"Order {order_id} modified successfully.")
        except Exception as e:
            logger.error(f"Error modifying order: {str(e)}")
            raise

    def cancel_order(self, order_id):
        try:
            self.kite.cancel_order(
                variety=self.kite.VARIETY_REGULAR,
                order_id=order_id
            )
            logger.info(f"Order {order_id} cancelled successfully.")
        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            raise

    def get_orders(self):
        try:
            return self.kite.orders()
        except Exception as e:
            logger.error(f"Error fetching orders: {str(e)}")
            raise

    def get_positions(self):
        try:
            return self.kite.positions()
        except Exception as e:
            logger.error(f"Error fetching positions: {str(e)}")
            raise

    def get_holdings(self):
        try:
            return self.kite.holdings()
        except Exception as e:
            logger.error(f"Error fetching holdings: {str(e)}")
            raise

    def get_margins(self):
        try:
            return self.kite.margins()
        except Exception as e:
            logger.error(f"Error fetching margins: {str(e)}")
            raise

class LiveDataManager:
    def __init__(self, kite: KiteApp):
        self.kite = kite
        self.subscribed_tokens = set()
        self.is_running = False
        self.thread = None

    def on_ticks(self, ticks):
        for tick in ticks:
            logger.info(f"Tick for {tick['instrument_token']}: LTP = {tick['last_price']}")

    def subscribe_symbols(self, symbols):
        try:
            instrument_tokens = []
            for symbol in symbols:
                token = self.get_instrument_token(symbol)
                if token:
                    instrument_tokens.append(token)
                    self.subscribed_tokens.add(token)

            if instrument_tokens:
                logger.info(f"Subscribed to: {', '.join(symbols)}")
            else:
                logger.warning("No valid symbols to subscribe.")
        except Exception as e:
            logger.error(f"Error subscribing to symbols: {str(e)}")
            raise

    def unsubscribe_symbols(self, symbols):
        try:
            for symbol in symbols:
                token = self.get_instrument_token(symbol)
                if token in self.subscribed_tokens:
                    self.subscribed_tokens.remove(token)

            logger.info(f"Unsubscribed from: {', '.join(symbols)}")
        except Exception as e:
            logger.error(f"Error unsubscribing from symbols: {str(e)}")
            raise

    def get_instrument_token(self, symbol):
        try:
            instruments = self.kite.instruments("NSE")
            for instrument in instruments:
                if instrument['tradingsymbol'] == symbol:
                    return instrument['instrument_token']
            logger.warning(f"Instrument token not found for {symbol}")
            return None
        except Exception as e:
            logger.error(f"Error fetching instrument token: {str(e)}")
            raise

    def is_market_open(self):
        india_tz = pytz.timezone('Asia/Kolkata')
        current_time = datetime.now(india_tz).time()
        market_open = dt_time(9, 15)
        market_close = dt_time(15, 30)
        return market_open <= current_time <= market_close

    def display_easter_egg(self):
        ascii_art = """
        Market's closed! The bull is sleeping...
             _____
            /     \\
           /       \\
          /  ^   ^  \\
         |  (o) (o)  |
          \\    <    /
           \\  ===  /
            \\_____/
              @brownbobdowney, here you go!!
        """
        quotes = [
            "The stock market is filled with individuals who know the price of everything, but the value of nothing. - Phillip Fisher",
            "The best way to measure your investing success is not by whether you're beating the market but by whether you've put in place a financial plan and a behavioral discipline that are likely to get you where you want to go. - Benjamin Graham",
            "In the short run, the market is a voting machine. In the long run, it's a weighing machine. - Benjamin Graham",
            "The stock market is a device for transferring money from the impatient to the patient. - Warren Buffett",
            "Be fearful when others are greedy and greedy when others are fearful. - Warren Buffett"
        ]
        print(ascii_art)
        print(random.choice(quotes))
        print("\n@brownbobdowney here you go!")

    def start_ticker(self):
        if not self.is_market_open():
            logger.warning("Market is currently closed. Live data stream not started.")
            self.display_easter_egg()
            return

        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._run_ticker)
            self.thread.start()
            logger.info("Live data stream started.")
        else:
            logger.warning("Ticker is already running.")

    def stop_ticker(self):
        if self.is_running:
            self.is_running = False
            if self.thread:
                self.thread.join()
            logger.info("Live data stream stopped.")
        else:
            logger.warning("Ticker is not running.")

    def _run_ticker(self):
        while self.is_running:
            try:
                if not self.is_market_open():
                    logger.info("Market has closed. Stopping live data stream.")
                    self.stop_ticker()
                    break

                if self.subscribed_tokens:
                    quotes = self.kite.quote(list(self.subscribed_tokens))
                    self.on_ticks([{'instrument_token': k, **v} for k, v in quotes.items()])
                time.sleep(1)  # Adjust the interval as needed
            except Exception as e:
                logger.error(f"Error in ticker: {str(e)}")
                time.sleep(5)  # Wait before retrying

def create_kite_app(enctoken):
    try:
        return KiteApp(enctoken=enctoken)
    except Exception as e:
        logger.error(f"Error creating KiteApp instance: {str(e)}")
        raise

# Example usage (for testing purposes)
if __name__ == "__main__":
    enctoken = input("Enter your Zerodha encryption token: ")
    
    try:
        kite = create_kite_app(enctoken)
        order_manager = OrderManager(kite)
        live_data_manager = LiveDataManager(kite)
        
        # Test order placement
        order_id = order_manager.place_order("NSE", "INFY", "BUY", 1, "CNC", "MARKET")
        print(f"Placed order: {order_id}")
        
        # Test fetching orders
        orders = order_manager.get_orders()
        print(f"Current orders: {orders}")
        
        # Test live data subscription
        live_data_manager.subscribe_symbols(["INFY", "TCS"])
        live_data_manager.start_ticker()
        
        # Keep the script running to receive ticks
        time.sleep(60)
        
        live_data_manager.stop_ticker()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
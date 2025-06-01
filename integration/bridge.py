import mmap
import struct
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import time
import os

class SharedMemoryBridge:
    def __init__(self, shm_name: str = "arbitragex_shm", size: int = 1024 * 1024):
        """
        Initialize shared memory bridge.
        
        Args:
            shm_name: Name of shared memory segment
            size: Size of shared memory in bytes
        """
        self.shm_name = shm_name
        self.size = size
        self.shm = None
        self.buffer = None
        
    def connect(self):
        """Connect to shared memory segment."""
        try:
            # Open shared memory file
            fd = os.open(f"/dev/shm/{self.shm_name}", os.O_RDWR)
            
            # Map shared memory
            self.shm = mmap.mmap(fd, self.size)
            self.buffer = np.frombuffer(self.shm, dtype=np.float64)
            
            print(f"Connected to shared memory: {self.shm_name}")
        except Exception as e:
            print(f"Error connecting to shared memory: {e}")
            raise
            
    def disconnect(self):
        """Disconnect from shared memory."""
        if self.shm:
            self.shm.close()
            self.shm = None
            self.buffer = None
            
    def write_order(self, order: Dict):
        """
        Write order to shared memory.
        
        Args:
            order: Dictionary containing order information
        """
        if not self.buffer:
            raise RuntimeError("Not connected to shared memory")
            
        # Pack order data
        data = struct.pack('!dii', 
            order['price'],
            order['quantity'],
            1 if order['is_buy'] else 0
        )
        
        # Write to shared memory
        self.shm.seek(0)
        self.shm.write(data)
        
    def read_market_data(self) -> Dict:
        """
        Read market data from shared memory.
        
        Returns:
            Dictionary containing market data
        """
        if not self.buffer:
            raise RuntimeError("Not connected to shared memory")
            
        # Read from shared memory
        self.shm.seek(0)
        data = self.shm.read(24)  # 3 doubles * 8 bytes
        
        # Unpack data
        best_bid, best_ask, last_price = struct.unpack('!ddd', data)
        
        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'last_price': last_price
        }

class OrderBookClient:
    def __init__(self, bridge: SharedMemoryBridge):
        """
        Initialize order book client.
        
        Args:
            bridge: Shared memory bridge instance
        """
        self.bridge = bridge
        
    def place_order(self, price: float, quantity: int, is_buy: bool):
        """
        Place an order through the bridge.
        
        Args:
            price: Order price
            quantity: Order quantity
            is_buy: True for buy order, False for sell
        """
        order = {
            'price': price,
            'quantity': quantity,
            'is_buy': is_buy
        }
        self.bridge.write_order(order)
        
    def get_market_data(self) -> Dict:
        """
        Get current market data.
        
        Returns:
            Dictionary containing market data
        """
        return self.bridge.read_market_data()

def main():
    # Example usage
    bridge = SharedMemoryBridge()
    client = OrderBookClient(bridge)
    
    try:
        # Connect to shared memory
        bridge.connect()
        
        # Place some orders
        client.place_order(100.0, 100, True)  # Buy 100 at 100.0
        time.sleep(0.1)
        client.place_order(101.0, 50, False)  # Sell 50 at 101.0
        
        # Get market data
        market_data = client.get_market_data()
        print("Market Data:", market_data)
        
    finally:
        # Clean up
        bridge.disconnect()

if __name__ == "__main__":
    main() 
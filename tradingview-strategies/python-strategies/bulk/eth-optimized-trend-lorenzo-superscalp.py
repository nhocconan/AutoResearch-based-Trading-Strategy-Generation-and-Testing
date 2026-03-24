#!/usr/bin/env python3
import numpy as np
import pandas as pd

name = "[ETH] Optimized Trend Strategy - Lorenzo SuperScalp"
timeframe = "1m"
leverage = 1

def calculate_rsi(close, length=14):
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Wilders smoothing (alpha = 1/length)
    # Implementing manually to ensure parity with Pine ta.rsi
    avg_gain = np.zeros_like(close, dtype=float)
    avg_loss = np.zeros_like(close, dtype=float)
    
    # Initial SMA for first valid point
    if len(close) > length:
        avg_gain[length] = np.mean(gain[:length+1])
        avg_loss[length] = np.mean(loss[:length+1])
        
        for i in range(length + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length - 1) + gain[i]) / length
            avg_loss[i] = (avg_loss[i-1] * (length - 1) + loss[i]) / length
    
    rs = np.zeros_like(close, dtype=float)
    mask = avg_loss != 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    # Handle initial NaNs
    rsi[:length] = np.nan
    return rsi

def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_macd(close, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    return macd_line, signal_line

def calculate_bb(close, length=20, mult=2.0):
    basis = close.rolling(window=length).mean()
    dev = close.rolling(window=length).std(ddof=1) * mult
    upper = basis + dev
    lower = basis - dev
    return basis, upper, lower

def generate_signals(prices):
    """
    Generates trading signals based on RSI, Bollinger Bands, and MACD.
    Returns a numpy array of target positions: 1.0 (Long), -1.0 (Short), 0.0 (Flat).
    Signals are shifted by 1 bar to prevent lookahead (decision at close i-1, execute open i).
    """
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("prices must be a pandas DataFrame")
    
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in prices.columns for col in required_cols):
        raise ValueError("Missing required columns")
    
    n = len(prices)
    if n == 0:
        return np.array([], dtype=float)
    
    close = prices['close'].astype(float)
    
    # Calculate Indicators
    rsi = calculate_rsi(close, 14)
    macd_line, signal_line = calculate_macd(close, 12, 26, 9)
    _, upper_band, lower_band = calculate_bb(close, 20, 2.0)
    
    # Initialize state
    signals = np.zeros(n, dtype=float)
    current_position = 0.0 # 0: Flat, 1: Long, -1: Short
    last_signal_buy = None # None, True, False
    last_trade_bar = -100 # Index of last trade decision
    min_bars_between_trades = 15
    
    # Loop starting from 1 to ensure previous data exists for decision
    # Signal[i] is decided using data[i-1] and executed at open[i]
    for i in range(1, n):
        # Use data from i-1 for decision making (close of previous bar)
        idx = i - 1
        
        # Check for NaNs in indicators
        if np.isnan(rsi[idx]) or np.isnan(macd_line[idx]) or np.isnan(signal_line[idx]) or np.isnan(upper_band[idx]) or np.isnan(lower_band[idx]):
            signals[i] = current_position
            continue
        
        # MACD Cross Conditions
        # Need previous bar macd data for crossover detection
        if idx == 0:
            macd_cross_up = False
            macd_cross_down = False
        else:
            prev_macd = macd_line[idx-1]
            prev_sig = signal_line[idx-1]
            curr_macd = macd_line[idx]
            curr_sig = signal_line[idx]
            
            macd_cross_up = (curr_macd > curr_sig) and (prev_macd <= prev_sig)
            macd_cross_down = (curr_macd < curr_sig) and (prev_macd >= prev_sig)
        
        # Signal Conditions
        rsi_val = rsi[idx]
        close_val = close[idx]
        upper_val = upper_band[idx]
        lower_val = lower_band[idx]
        
        buy_condition = (rsi_val < 45) and (close_val < lower_val * 1.02) and macd_cross_up
        sell_condition = (rsi_val > 55) and (close_val > upper_val * 0.98) and macd_cross_down
        
        # Cooldown Check
        time_elapsed = (idx - last_trade_bar) >= min_bars_between_trades
        
        # Logic State Checks
        can_buy = False
        can_sell = False
        
        if buy_condition:
            if last_signal_buy is None or last_signal_buy is False:
                if time_elapsed:
                    can_buy = True
        
        if sell_condition:
            if last_signal_buy is True:
                if time_elapsed:
                    can_sell = True
        
        # Execute Logic
        if can_buy:
            current_position = 1.0
            last_signal_buy = True
            last_trade_bar = idx
        elif can_sell:
            current_position = -1.0
            last_signal_buy = False
            last_trade_bar = idx
        # Else: maintain current_position (Hold or Flat)
        
        signals[i] = current_position
    
    return signals

if __name__ == "__main__":
    # Example usage for validation
    pass

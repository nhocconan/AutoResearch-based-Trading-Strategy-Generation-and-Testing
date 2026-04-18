#!/usr/bin/env python3
"""
1d_RSI_MeanReversion_SmallCap
Hypothesis: On the daily timeframe, extreme RSI values indicate exhaustion in BTC/ETH mean-reverting opportunities. 
Enter long when RSI(14) < 30 and price > 200-day SMA (avoid dead cats), short when RSI(14) > 70 and price < 200-day SMA.
Volume must be > 1.5x 20-day average to confirm participation. 
Exit when RSI returns to neutral range (40-60) or opposite extreme is reached.
Designed for low frequency (~10-20 trades/year) to minimize fee drag in choppy markets like 2025.
"""

import numpy as np
import pandas as pd
from typing import Tuple

def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI with Wilder's smoothing."""
    delta = np.diff(close)
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    
    roll_up = np.zeros_like(close)
    roll_down = np.zeros_like(close)
    
    if len(close) >= period:
        roll_up[period] = np.mean(up[:period])
        roll_down[period] = np.mean(down[:period])
        for i in range(period + 1, len(close)):
            roll_up[i] = (roll_up[i-1] * (period - 1) + up[i-1]) / period
            roll_down[i] = (roll_down[i-1] * (period - 1) + down[i-1]) / period
    
    rs = np.where(roll_down != 0, roll_up / roll_down, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_sma(arr: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average."""
    sma = np.full_like(arr, np.nan)
    for i in range(period - 1, len(arr)):
        sma[i] = np.mean(arr[i - period + 1:i + 1])
    return sma

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Indicators
    rsi = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    
    # Volume confirmation: >1.5x 20-day average
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20)  # Need 200-day SMA and 20-day vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(rsi[i]) or np.isnan(sma_200[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI oversold + price above 200-day SMA + volume confirmation
            if rsi[i] < 30 and close[i] > sma_200[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought + price below 200-day SMA + volume confirmation
            elif rsi[i] > 70 and close[i] < sma_200[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral (40-60) or becomes overbought
            if rsi[i] >= 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral (40-60) or becomes oversold
            if rsi[i] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_RSI_MeanReversion_SmallCap"
timeframe = "1d"
leverage = 1.0
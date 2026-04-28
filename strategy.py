#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Pullback_Volume_Regime
Hypothesis: Use daily KAMA (2,30) for trend direction, RSI(14) pullback to 40/60 for entry, and volume > 1.5x 20-day average for confirmation. Exit when RSI crosses 50 opposite direction. Designed for low trade frequency (7-25/year) to minimize fee decay while capturing trend continuations in both bull and bear markets. KAMA adapts to market noise, reducing false signals during chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (2,30) on close
    # ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close, n=1))
    dir = np.abs(np.subtract(close[10:], close[:-10]))  # 10-period momentum
    vol = np.sum(change[10:], axis=0) if len(change) >= 10 else np.array([])
    # Pad arrays to match length
    if len(dir) < n - 9:
        dir = np.pad(dir, (0, n - 9 - len(dir)), 'constant', constant_values=np.nan)
    if len(vol) < n - 9:
        vol = np.pad(vol, (0, n - 9 - len(vol)), 'constant', constant_values=np.nan)
    er = np.full(n, np.nan)
    if len(vol) >= n - 9:
        er[9:] = np.where(vol[9:] != 0, dir[9:] / vol[9:], 0)
    # Smooth ER with smoothing constants
    sc = (er * 0.288 + 0.064) ** 2  # where 0.288 = 2/(2+1), 0.064 = 2/(30+1)
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad RSI to match length (first 14 values are NaN)
    rsi_padded = np.full(n, np.nan)
    rsi_padded[14:] = rsi
    
    # Volume confirmation: >1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Wait for KAMA and volume MA to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi_padded[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction: price vs KAMA
        uptrend = close[i] > kama[i]
        downtrend = close[i] < kama[i]
        
        # RSI pullback levels: long when RSI < 40 in uptrend, short when RSI > 60 in downtrend
        long_setup = rsi_padded[i] < 40 and uptrend
        short_setup = rsi_padded[i] > 60 and downtrend
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Entry conditions
        long_entry = long_setup and vol_confirm
        short_entry = short_setup and vol_confirm
        
        # Exit conditions: RSI crosses 50 in opposite direction
        long_exit = rsi_padded[i] > 50 and position == 1
        short_exit = rsi_padded[i] < 50 and position == -1
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_KAMA_Direction_RSI_Pullback_Volume_Regime"
timeframe = "1d"
leverage = 1.0
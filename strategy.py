#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based stop
# - Long when price breaks above 4h Donchian upper channel (20-period high) with volume > 1.8x 20-period average
# - Short when price breaks below 4h Donchian lower channel (20-period low) with volume > 1.8x 20-period average
# - Exit when price crosses back through the 10-period moving average (opposite side) or ATR stop hit (2.5x ATR)
# - Uses 4h for all calculations (no mixing timeframes for core logic)
# - Target: 20-40 trades per year per symbol (80-160 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 4h price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 10-period moving average for exit
    ma_10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop loss (20-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(ma_10[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price breaks above upper Donchian + volume surge
            if price > high_roll[i] and vol > 1.8 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below lower Donchian + volume surge
            elif price < low_roll[i] and vol > 1.8 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below 10-period MA OR ATR stop hit (2.5*ATR)
            if price < ma_10[i] or price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 10-period MA OR ATR stop hit (2.5*ATR)
            if price > ma_10[i] or price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSurge_MA10Exit_ATRStop"
timeframe = "4h"
leverage = 1.0
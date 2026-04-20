#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with volume confirmation and ATR stop
# - Uses 1d for Donchian channel calculation (stable levels)
# - Long when price breaks above upper band with volume > 1.5x 10-period average
# - Short when price breaks below lower band with volume > 1.5x 10-period average
# - Exit when price crosses back through midline or ATR-based stop hit
# - Target: 20-30 trades per year per symbol (80-120 total over 4 years)
# - Designed to work in both bull (breakout continuation) and bear (mean reversion via midline)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR for stop loss (using 1d data)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate Donchian channel from previous day
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    upper_band = pd.Series(high_prev).rolling(window=20, min_periods=20).max().values
    lower_band = pd.Series(low_prev).rolling(window=20, min_periods=20).min().values
    mid_band = (upper_band + lower_band) / 2
    
    # Align Donchian levels to 12h timeframe
    upper_12h = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_12h = align_htf_to_ltf(prices, df_1d, lower_band)
    mid_12h = align_htf_to_ltf(prices, df_1d, mid_band)
    
    # 12h price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 10-period average
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(upper_12h[i]) or np.isnan(lower_12h[i]) or np.isnan(mid_12h[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(atr_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price breaks above upper band + volume surge
            if price > upper_12h[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below lower band + volume surge
            elif price < lower_12h[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below mid band OR ATR stop hit (2*ATR)
            if price < mid_12h[i] or price < entry_price - 2.0 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above mid band OR ATR stop hit (2*ATR)
            if price > mid_12h[i] or price > entry_price + 2.0 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_Volume_ATRStop"
timeframe = "12h"
leverage = 1.0
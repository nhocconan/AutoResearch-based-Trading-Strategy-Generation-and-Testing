#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume confirmation and ATR stop
# - Use 12h timeframe for execution
# - Calculate Donchian(20) channels from daily data (stable levels)
# - Long when price breaks above upper band with volume > 1.5x 20-period average
# - Short when price breaks below lower band with volume > 1.5x 20-period average
# - Exit when price crosses back through opposite band or ATR-based stop hit
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)
# - Works in both bull/bear: breakouts capture momentum, ATR stop limits downside

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian(20) channels from daily data
    high_max = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    upper_12h = align_htf_to_ltf(prices, df_1d, high_max)
    lower_12h = align_htf_to_ltf(prices, df_1d, low_min)
    
    # Calculate ATR for stop loss (using 1d data)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 12h price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(upper_12h[i]) or np.isnan(lower_12h[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(atr_12h[i]):
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
            # Long exit: price crosses below lower band OR ATR stop hit (2*ATR)
            if price < lower_12h[i] or price < entry_price - 2.0 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above upper band OR ATR stop hit (2*ATR)
            if price > upper_12h[i] or price > entry_price + 2.0 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Volume_ATRStop"
timeframe = "12h"
leverage = 1.0
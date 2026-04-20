#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation
# - Calculate weekly Donchian channels (20-period high/low)
# - Long when price breaks above weekly upper band with volume > 1.5x 20-day average
# - Short when price breaks below weekly lower band with volume > 1.5x 20-day average
# - Exit when price crosses back through the opposite weekly band (e.g., long exits at lower band)
# - Uses 1d for execution and 1w for trend/context (stable levels)
# - Target: 15-25 trades per year per symbol (60-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for Donchian channel calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR for stop loss (using weekly data)
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate weekly Donchian channels (20-period)
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    upper_1w = align_htf_to_ltf(prices, df_1w, high_20)
    lower_1w = align_htf_to_ltf(prices, df_1w, low_20)
    
    # 1d price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(upper_1w[i]) or np.isnan(lower_1w[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(atr_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price breaks above weekly upper band + volume surge
            if price > upper_1w[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below weekly lower band + volume surge
            elif price < lower_1w[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below weekly lower band OR ATR stop hit (2*ATR)
            if price < lower_1w[i] or price < entry_price - 2.0 * atr_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly upper band OR ATR stop hit (2*ATR)
            if price > upper_1w[i] or price > entry_price + 2.0 * atr_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wTrend_Volume_ATRStop"
timeframe = "1d"
leverage = 1.0
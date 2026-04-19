#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian(20) breakout with weekly EMA200 trend filter and volume spike confirmation.
# Long when: Price breaks above 20-day high, weekly EMA200 upward, volume > 1.5x 20-day average volume
# Short when: Price breaks below 20-day low, weekly EMA200 downward, volume > 1.5x 20-day average volume
# Exit when: Price crosses back through the 20-day midpoint (mean of 20-day high and low)
# Works in bull (buy breakouts) and bear (sell breakdowns) due to trend filter.
# Target: 15-25 trades/year per symbol. Low frequency minimizes fee drag.
name = "1d_Donchian20_EMA200_Volume_Spike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 20-day Donchian channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2.0  # Exit level
    
    # 20-day average volume for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Weekly data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for EMA200 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(mid_20[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(ema200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        high20 = high_20[i]
        low20 = low_20[i]
        mid20 = mid_20[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema200 = ema200_1w_aligned[i]
        
        if position == 0:
            # Long entry: Price breaks above 20-day high, EMA200 upward, volume spike
            if (price > high20 and close[i-1] <= high20 and 
                ema200 > ema200_1w_aligned[i-1] and vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below 20-day low, EMA200 downward, volume spike
            elif (price < low20 and close[i-1] >= low20 and 
                  ema200 < ema200_1w_aligned[i-1] and vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses back below 20-day midpoint
            if price < mid20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses back above 20-day midpoint
            if price > mid20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
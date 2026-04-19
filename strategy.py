#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w high/low trend filter and volume confirmation
# In trending markets, price breaks Donchian channels with volume (breakout)
# Uses 1w high/low as trend filter: only long when above 1w low, short when below 1w high
# Volume confirmation ensures breakouts are genuine
# Works in both bull and bear markets by adapting to volatility regime via breakout strength
# Target: 12-37 trades/year per symbol (~50-150 total over 4 years)

name = "12h_Donchian20_1wTrend_Filter_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian(20) on 12h
    highest_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w high/low for trend filter
    highest_1w = pd.Series(high_1w).rolling(window=5, min_periods=5).max().values  # ~5 weeks
    lowest_1w = pd.Series(low_1w).rolling(window=5, min_periods=5).min().values
    
    # Align indicators to 12h timeframe
    highest_20_aligned = align_htf_to_ltf(prices, df_12h, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_12h, lowest_20)
    highest_1w_aligned = align_htf_to_ltf(prices, df_1w, highest_1w)
    lowest_1w_aligned = align_htf_to_ltf(prices, df_1w, lowest_1w)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need Donchian and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or 
            np.isnan(highest_1w_aligned[i]) or np.isnan(lowest_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume and trend filters
        volume_confirmed = vol > 2.0 * vol_ma
        uptrend = price > lowest_1w_aligned[i]  # Above weekly low = bullish bias
        downtrend = price < highest_1w_aligned[i]  # Below weekly high = bearish bias
        
        # Donchian levels
        upper = highest_20_aligned[i]
        lower = lowest_20_aligned[i]
        
        if position == 0:
            # Long: breakout above upper Donchian with volume and uptrend bias
            if price > upper and volume_confirmed and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below lower Donchian with volume and downtrend bias
            elif price < lower and volume_confirmed and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to midpoint of Donchian channel
            midpoint = (upper + lower) * 0.5
            if price < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to midpoint of Donchian channel
            midpoint = (upper + lower) * 0.5
            if price > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
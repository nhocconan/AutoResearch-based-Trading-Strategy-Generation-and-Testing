#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with weekly trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) AND weekly EMA20 rising AND volume > 1.3x 20-period average.
# Short when Williams %R > -20 (overbought) AND weekly EMA20 falling AND volume > 1.3x 20-period average.
# Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts).
# Williams %R identifies overextended moves. Weekly EMA20 filters higher timeframe trend.
# Volume filter confirms institutional participation. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_WilliamsR_WeeklyEMA20_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Weekly EMA20 for trend filter
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Weekly EMA20 direction
    ema20_rising = np.zeros_like(ema20_1w_aligned, dtype=bool)
    ema20_falling = np.zeros_like(ema20_1w_aligned, dtype=bool)
    ema20_rising[1:] = ema20_1w_aligned[1:] > ema20_1w_aligned[:-1]
    ema20_falling[1:] = ema20_1w_aligned[1:] < ema20_1w_aligned[:-1]
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Sufficient warmup for Williams %R and EMA20
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(ema20_rising[i]) or np.isnan(ema20_falling[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R oversold (< -80), weekly EMA20 rising, volume filter
            long_cond = (williams_r[i] < -80) and ema20_rising[i] and volume_filter[i]
            # Short conditions: Williams %R overbought (> -20), weekly EMA20 falling, volume filter
            short_cond = (williams_r[i] > -20) and ema20_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses back above -50
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses back below -50
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
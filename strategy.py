#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and volume spike confirmation.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Long when Bull Power > 0, 1d EMA50 rising, and volume > 2x 20-period average.
# Short when Bear Power > 0, 1d EMA50 falling, and volume > 2x 20-period average.
# Exit when power crosses zero (Bull Power <= 0 for long exit, Bear Power <= 0 for short exit).
# Elder Ray measures bull/bear strength relative to EMA13. EMA50 filters higher timeframe trend.
# Volume spike confirms institutional participation. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_ElderRay_1dEMA50_Volume"
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
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # EMA13 for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d EMA50 direction
    ema50_rising = np.zeros_like(ema50_1d_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_1d_aligned, dtype=bool)
    ema50_rising[1:] = ema50_1d_aligned[1:] > ema50_1d_aligned[:-1]
    ema50_falling[1:] = ema50_1d_aligned[1:] < ema50_1d_aligned[:-1]
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # Sufficient warmup for EMA13 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_rising[i]) or 
            np.isnan(ema50_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0, 1d EMA50 rising, volume filter
            long_cond = (bull_power[i] > 0) and ema50_rising[i] and volume_filter[i]
            # Short conditions: Bear Power > 0, 1d EMA50 falling, volume filter
            short_cond = (bear_power[i] > 0) and ema50_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 (bullish momentum faded)
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power <= 0 (bearish momentum faded)
            if bear_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray (Bull/Bear Power) with weekly trend filter and volume confirmation
# Long when Bull Power > 0, weekly EMA(34) rising, and volume spike
# Short when Bear Power < 0, weekly EMA(34) falling, and volume spike
# Uses 6h timeframe to target 12-37 trades/year, avoiding excessive frequency.
# Elder Ray measures bull/bear power relative to EMA, providing directional momentum.
# Weekly trend filter ensures alignment with higher timeframe momentum.
# Volume spike confirms institutional participation.

name = "6h_ElderRay_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    weekly_close = df_1w['close'].values
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_slope = np.diff(ema34_1w, prepend=ema34_1w[0])  # slope for trend direction
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    ema34_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w_slope)
    
    # Calculate EMA(13) for Elder Ray (6h timeframe)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA
    bear_power = low - ema13   # Bear Power: Low - EMA
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(ema34_1w_slope_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1w_val = ema34_1w_aligned[i]
        ema34_1w_slope_val = ema34_1w_slope_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Bull Power > 0, weekly EMA rising, volume spike
            if bull_val > 0 and ema34_1w_slope_val > 0 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power < 0, weekly EMA falling, volume spike
            elif bear_val < 0 and ema34_1w_slope_val < 0 and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 OR weekly EMA starts falling
            if bull_val <= 0 or ema34_1w_slope_val <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 OR weekly EMA starts rising
            if bear_val >= 0 or ema34_1w_slope_val >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
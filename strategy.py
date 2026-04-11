#!/usr/bin/env python3
# 6h_1d_elder_ray_zone_v1
# Strategy: Elder Ray Power Zone with 1d trend filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) identifies
# strong/weak price action relative to trend. Trade when both powers align with 1d EMA trend
# and volume confirms. Works in bull via bull power strength, in bear via bear power weakness.
# Low turnover expected due to dual confirmation requirements.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_elder_ray_zone_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily EMA13 for trend filter
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_trend = ema13_1d > np.roll(ema13_1d, 1)  # Rising EMA13 = uptrend
    ema13_trend[0] = False  # First value invalid
    ema13_trend_aligned = align_htf_to_ltf(prices, df_1d, ema13_trend)
    
    # 6h EMA13 for Elder Ray calculation
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Power
    bull_power = high - ema13_6h  # Strength above trend
    bear_power = ema13_6h - low   # Weakness below trend
    
    # Smooth powers (3-period) to reduce noise
    bull_power_smooth = pd.Series(bull_power).ewm(span=3, adjust=False, min_periods=3).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(ema13_trend_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions
        long_entry = (bull_power_smooth[i] > 0 and  # Bull power positive
                     bear_power_smooth[i] < 0 and   # Bear power negative (confirms trend)
                     ema13_trend_aligned[i] and     # 1d uptrend
                     vol_spike[i])                  # Volume confirmation
        
        short_entry = (bull_power_smooth[i] < 0 and  # Bull power negative
                      bear_power_smooth[i] > 0 and   # Bear power positive
                      not ema13_trend_aligned[i] and # 1d downtrend
                      vol_spike[i])                  # Volume confirmation
        
        # Exit conditions: power divergence or trend change
        exit_long = position == 1 and (bull_power_smooth[i] <= 0 or not ema13_trend_aligned[i])
        exit_short = position == -1 and (bear_power_smooth[i] <= 0 or ema13_trend_aligned[i])
        
        # Trading logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
#!/usr/bin/env python3
"""
#100943 - 4h_Camarilla_R1_S1_Breakout_12hTrend_Volume
Hypothesis: Breakout at Camarilla R1/S1 levels with volume confirmation and 12h EMA trend filter on 4h timeframe. Targets 20-50 trades/year to minimize fee drag. Uses 4h primary timeframe with 12h EMA50 for trend and daily pivot for Camarilla levels. Works in bull (breakouts with trend) and bear (mean reversion to pivot). Tight entry conditions reduce trade frequency and improve win rate.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Get 1d data for Camarilla levels (daily pivot from previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day (to avoid look-ahead)
    daily_pivot = (high_1d + low_1d + close_1d) / 3
    daily_range = high_1d - low_1d
    daily_r1 = close_1d + daily_range * 1.1 / 12
    daily_s1 = close_1d - daily_range * 1.1 / 12
    
    # Align to 4h timeframe (previous day's levels for current period)
    camarilla_r1 = align_htf_to_ltf(prices, df_1d, daily_r1)
    camarilla_s1 = align_htf_to_ltf(prices, df_1d, daily_s1)
    camarilla_pivot = align_htf_to_ltf(prices, df_1d, daily_pivot)
    
    # Volume filter: volume > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(camarilla_r1[i]) or 
            np.isnan(camarilla_s1[i]) or np.isnan(camarilla_pivot[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above R1, above 12h EMA50, volume spike
        if (close[i] > camarilla_r1[i] and 
            close[i] > ema50_12h_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below S1, below 12h EMA50, volume spike
        elif (close[i] < camarilla_s1[i] and 
              close[i] < ema50_12h_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to Camarilla Pivot (mean reversion)
        elif position == 1 and close[i] < camarilla_pivot[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > camarilla_pivot[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
# Hypothesis: Camarilla pivot levels (R1/S1) from daily timeframe act as strong support/resistance.
# Breakouts above R1 or below S1 with volume confirmation and daily trend filter capture
# continuation moves. Works in bull markets (buy R1 breakouts in uptrend) and bear markets
# (sell S1 breakdowns in downtrend). Uses 12h timeframe to limit trade frequency and
# reduce fee drag, targeting 50-150 total trades over 4 years.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # Avoid division by zero and handle NaN
    hl_range = high_1d - low_1d
    # Only calculate where range is valid
    valid_range = hl_range > 0
    camarilla_width = np.zeros_like(hl_range)
    camarilla_width[valid_range] = 1.1 * hl_range[valid_range] / 12.0
    
    r1 = close_1d_arr + camarilla_width
    s1 = close_1d_arr - camarilla_width
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation (2-period MA on 12h chart = 1 day)
    volume_ma = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50_1d (50), volume MA (2)
    start_idx = max(50, 2)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: uptrend + close breaks above R1 + volume
            if uptrend and close[i] > r1_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + close breaks below S1 + volume
            elif downtrend and close[i] < s1_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters below R1
            if not uptrend or close[i] < r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price re-enters above S1
            if not downtrend or close[i] > s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
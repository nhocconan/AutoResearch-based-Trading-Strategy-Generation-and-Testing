#!/usr/bin/env python3
# 4H_1D_1W_Camarilla_R1S1_Breakout_VolumeSpike_TrendFilter
# Hypothesis: On 4h timeframe, enter long when price breaks above Camarilla R1 level during a daily uptrend (EMA34) with volume confirmation (>1.5x 20-period average).
# Enter short when price breaks below Camarilla S1 level during a daily downtrend with volume confirmation.
# Use weekly trend filter to avoid counter-trend trades in strong weekly trends.
# Weekly trend filter: price above/below weekly EMA34.
# Target: 20-50 trades/year per symbol (80-200 total over 4 years).

name = "4H_1D_1W_Camarilla_R1S1_Breakout_VolumeSpike_TrendFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R1, S1) from previous day
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Get weekly data for trend filter (EMA34 on weekly close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    weekly_ema34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Daily trend filter: EMA34 on daily close
    daily_ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    # Align Camarilla levels to 4h
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Align daily EMA34 to 4h
    daily_ema34_aligned = align_htf_to_ltf(prices, df_1d, daily_ema34)
    
    # Align weekly EMA34 to 4h
    weekly_ema34_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or \
           np.isnan(daily_ema34_aligned[i]) or np.isnan(weekly_ema34_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Camarilla R1 + daily uptrend (price > EMA34) + weekly uptrend + volume confirmation
            if close[i] > camarilla_r1_aligned[i] and close[i] > daily_ema34_aligned[i] and close[i] > weekly_ema34_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Camarilla S1 + daily downtrend (price < EMA34) + weekly downtrend + volume confirmation
            elif close[i] < camarilla_s1_aligned[i] and close[i] < daily_ema34_aligned[i] and close[i] < weekly_ema34_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Camarilla S1 or weekly downtrend
            if close[i] < camarilla_s1_aligned[i] or close[i] < weekly_ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Camarilla R1 or weekly uptrend
            if close[i] > camarilla_r1_aligned[i] or close[i] > weekly_ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
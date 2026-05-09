#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: On 4h timeframe, go long when weekly trend is up and price breaks above R1 with volume > 1.5x average.
# Go short when weekly trend is down and price breaks below S1 with volume > 1.5x average.
# Uses weekly trend filter to capture major trends and reduce whipsaw in both bull and bear markets.
# Tight entry conditions to limit trades and avoid fee drag.

name = "4h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "4h"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34 for trend filter
    ema34_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 34:
        ema34_1w[33] = np.mean(close_1w[0:34])
        for i in range(34, len(close_1w)):
            ema34_1w[i] = (close_1w[i] * 2 + ema34_1w[i-1] * 32) / 34
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels (R1, S1) from previous day
    camarilla_r1 = np.full_like(high_1d, np.nan)
    camarilla_s1 = np.full_like(low_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        # Use previous day's data
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        camarilla_r1[i] = pc + (ph - pl) * 1.1 / 12
        camarilla_s1[i] = pc - (ph - pl) * 1.1 / 12
    
    # Align weekly trend and daily Camarilla to 4h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume filter: current volume vs 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 1, 20)  # Need weekly EMA, daily Camarilla, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        trend_up = close[i] > ema34_1w_aligned[i]
        
        if position == 0:
            # Enter long: weekly trend up + price breaks above R1 + volume confirmation
            if trend_up and close[i] > camarilla_r1_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly trend down + price breaks below S1 + volume confirmation
            elif not trend_up and close[i] < camarilla_s1_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly trend turns down or price breaks below S1
            if not trend_up or close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly trend turns up or price breaks above R1
            if trend_up or close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
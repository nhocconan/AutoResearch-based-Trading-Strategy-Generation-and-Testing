#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_Filter
Hypothesis: Trade breakouts at daily Camarilla R1/S1 levels with volume confirmation on 12h timeframe.
Long when price breaks above R1 with volume spike; short when price breaks below S1 with volume spike.
Exit when price returns to daily pivot point (PP). Uses weekly trend filter to avoid counter-trend trades.
Camarilla levels provide institutional reference points; volume confirms institutional participation.
Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag.
Works in bull/bear: weekly trend filter ensures trades align with higher timeframe trend.
"""

name = "12h_Camarilla_R1_S1_Breakout_Volume_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from daily OHLC
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # PP = (H+L+C)/3
    H_daily = df_daily['high'].values
    L_daily = df_daily['low'].values
    C_daily = df_daily['close'].values
    
    R1_daily = C_daily + (H_daily - L_daily) * 1.1 / 12
    S1_daily = C_daily - (H_daily - L_daily) * 1.1 / 12
    PP_daily = (H_daily + L_daily + C_daily) / 3.0
    
    # Align daily levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_daily, R1_daily)
    S1_aligned = align_htf_to_ltf(prices, df_daily, S1_daily)
    PP_aligned = align_htf_to_ltf(prices, df_daily, PP_daily)
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_weekly = df_weekly['close'].values
    ema20_weekly = np.full_like(close_weekly, np.nan)
    if len(close_weekly) >= 20:
        multiplier = 2.0 / (20 + 1)
        ema20_weekly[19] = np.mean(close_weekly[:20])
        for i in range(20, len(close_weekly)):
            ema20_weekly[i] = multiplier * close_weekly[i] + (1 - multiplier) * ema20_weekly[i-1]
    ema20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, n):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(PP_aligned[i]) or np.isnan(ema20_weekly_aligned[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + volume spike + weekly uptrend
            if close[i] > R1_aligned[i] and volume_filter[i] and close[i] > ema20_weekly_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume spike + weekly downtrend
            elif close[i] < S1_aligned[i] and volume_filter[i] and close[i] < ema20_weekly_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot OR weekly trend turns down
            if close[i] < PP_aligned[i] or close[i] < ema20_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot OR weekly trend turns up
            if close[i] > PP_aligned[i] or close[i] > ema20_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
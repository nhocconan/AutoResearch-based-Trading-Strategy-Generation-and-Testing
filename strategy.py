#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Refined
# Hypothesis: Refined version with tighter entry conditions (volume > 2x average) and reduced position size (0.20) to lower trade frequency and improve generalization.
# Long when 1d trend up (price > EMA34) and price breaks above R3 with volume > 2x average.
# Short when 1d trend down (price < EMA34) and price breaks below S3 with volume > 2x average.
# Uses volume confirmation to filter breakouts and reduce false signals. Position size 0.20 to manage risk.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Refined"
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
    
    # Get 1d data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[0:34])
        for i in range(34, len(close_1d)):
            ema34_1d[i] = (close_1d[i] * 2 + ema34_1d[i-1] * 32) / 34
    
    # Calculate Camarilla levels (R3, S3) from previous day
    camarilla_r3 = np.full_like(high_1d, np.nan)
    camarilla_s3 = np.full_like(low_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        # Use previous day's data
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        camarilla_r3[i] = pc + (ph - pl) * 1.1 / 4
        camarilla_s3[i] = pc - (ph - pl) * 1.1 / 4
    
    # Align 1d indicators to 12h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
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
    
    start_idx = max(34, 1, 20)  # Need 1d EMA, Camarilla, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        trend_up = close[i] > ema34_1d_aligned[i]
        
        if position == 0:
            # Enter long: 1d trend up + price breaks above R3 + volume confirmation (tighter: >2x)
            if trend_up and close[i] > camarilla_r3_aligned[i] and volume_ratio[i] > 2.0:
                signals[i] = 0.20
                position = 1
            # Enter short: 1d trend down + price breaks below S3 + volume confirmation (tighter: >2x)
            elif not trend_up and close[i] < camarilla_s3_aligned[i] and volume_ratio[i] > 2.0:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: 1d trend turns down or price breaks below S3
            if not trend_up or close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: 1d trend turns up or price breaks above R3
            if trend_up or close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
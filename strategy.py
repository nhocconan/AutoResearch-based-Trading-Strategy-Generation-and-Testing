#!/usr/bin/env python3
# 4h_12h_Camarilla_R3_S3_Breakout_12hTrend_Volume
# Hypothesis: Camarilla R3/S3 breakout on 4h with 12h EMA trend filter and volume confirmation.
# Long when 12h trend up and price breaks above R3 with volume > 1.5x average.
# Short when 12h trend down and price breaks below S3 with volume > 1.5x average.
# Uses 12h timeframe for trend and structure, 4h for entry timing to balance signal frequency and avoid overtrading.

name = "4h_12h_Camarilla_R3_S3_Breakout_12hTrend_Volume"
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
    
    # Get 12h data for trend filter and Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h EMA34 for trend filter
    ema34_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 34:
        ema34_12h[33] = np.mean(close_12h[0:34])
        for i in range(34, len(close_12h)):
            ema34_12h[i] = (close_12h[i] * 2 + ema34_12h[i-1] * 32) / 34
    
    # Calculate Camarilla levels (R3, S3) from previous 12h bar
    camarilla_r3_12h = np.full_like(high_12h, np.nan)
    camarilla_s3_12h = np.full_like(low_12h, np.nan)
    
    for i in range(1, len(close_12h)):
        # Use previous 12h bar's data
        ph = high_12h[i-1]
        pl = low_12h[i-1]
        pc = close_12h[i-1]
        
        camarilla_r3_12h[i] = pc + (ph - pl) * 1.1 / 4
        camarilla_s3_12h[i] = pc - (ph - pl) * 1.1 / 4
    
    # Align 12h indicators to 4h timeframe
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    camarilla_r3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3_12h)
    camarilla_s3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3_12h)
    
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
    
    start_idx = max(34, 1, 20)  # Need 12h EMA, Camarilla, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(camarilla_r3_12h_aligned[i]) or 
            np.isnan(camarilla_s3_12h_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 12h trend
        trend_up = close[i] > ema34_12h_aligned[i]
        
        if position == 0:
            # Enter long: 12h trend up + price breaks above R3 + volume confirmation
            if trend_up and close[i] > camarilla_r3_12h_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: 12h trend down + price breaks below S3 + volume confirmation
            elif not trend_up and close[i] < camarilla_s3_12h_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: 12h trend turns down or price breaks below S3
            if not trend_up or close[i] < camarilla_s3_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: 12h trend turns up or price breaks above R3
            if trend_up or close[i] > camarilla_r3_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
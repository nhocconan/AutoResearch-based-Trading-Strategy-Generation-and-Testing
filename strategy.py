#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA20 trend filter and volume confirmation.
# Uses 4h trend for direction, 1h for precise entry timing, volume filter to avoid false breakouts.
# Designed for 1h timeframe to generate ~15-35 trades/year (~60-140 total over 4 years) to avoid fee drag.
# Long when 4h trend up (close > EMA20), price breaks above R1, volume > 1.8x average.
# Short when 4h trend down (close < EMA20), price breaks below S1, volume > 1.8x average.
# Session filter (08-20 UTC) to avoid low-volume Asian session noise.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
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
    
    # Get 4h data for trend filter and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h EMA20 for trend filter
    ema20_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 20:
        ema20_4h[19] = np.mean(close_4h[0:20])
        for i in range(20, len(close_4h)):
            ema20_4h[i] = (close_4h[i] * 2 + ema20_4h[i-1] * 18) / 20
    
    # Align 4h EMA20 to 1h timeframe
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Calculate Camarilla levels for each 4h bar: R1, S1
    camarilla_r1_4h = np.full_like(close_4h, np.nan)
    camarilla_s1_4h = np.full_like(close_4h, np.nan)
    
    for i in range(len(df_4h)):
        if not (np.isnan(high_4h[i]) or np.isnan(low_4h[i]) or np.isnan(close_4h[i])):
            camarilla_r1_4h[i] = close_4h[i] + 1.1 * (high_4h[i] - low_4h[i]) / 12
            camarilla_s1_4h[i] = close_4h[i] - 1.1 * (high_4h[i] - low_4h[i]) / 12
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1_4h)
    camarilla_s1_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1_4h)
    
    # Volume filter: current volume vs 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need 4h EMA20 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(camarilla_r1_4h_aligned[i]) or 
            np.isnan(camarilla_s1_4h_aligned[i]) or np.isnan(volume_ratio[i]) or
            not (8 <= hours[i] <= 20)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 4h trend
        trend_up = close[i] > ema20_4h_aligned[i]
        
        if position == 0:
            # Enter long: 4h trend up + price breaks above R1 + volume confirmation
            if trend_up and close[i] > camarilla_r1_4h_aligned[i] and volume_ratio[i] > 1.8:
                signals[i] = 0.20
                position = 1
            # Enter short: 4h trend down + price breaks below S1 + volume confirmation
            elif not trend_up and close[i] < camarilla_s1_4h_aligned[i] and volume_ratio[i] > 1.8:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: 4h trend turns down or price breaks below S1
            if not trend_up or close[i] < camarilla_s1_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: 4h trend turns up or price breaks above R1
            if trend_up or close[i] > camarilla_r1_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
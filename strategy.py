#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume confirmation.
# Uses daily trend to filter breakouts, reducing false signals in choppy markets.
# Designed to generate ~15-25 trades/year on 12h to avoid fee drag while maintaining edge.
# Long when 1d trend up (close > EMA50), price breaks above R1, volume > 1.5x average.
# Short when 1d trend down (close < EMA50), price breaks below S1, volume > 1.5x average.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema50_1d[i] = (close_1d[i] * 2 + ema50_1d[i-1] * 48) / 50
    
    # Align 1d EMA50 to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla levels for each 1d bar: R1, S1
    camarilla_r1_1d = np.full_like(close_1d, np.nan)
    camarilla_s1_1d = np.full_like(close_1d, np.nan)
    
    for i in range(len(df_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i])):
            camarilla_r1_1d[i] = close_1d[i] + 1.1 * (high_1d[i] - low_1d[i]) / 4
            camarilla_s1_1d[i] = close_1d[i] - 1.1 * (high_1d[i] - low_1d[i]) / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    
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
    
    start_idx = max(50, 20)  # Need 1d EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(camarilla_r1_1d_aligned[i]) or 
            np.isnan(camarilla_s1_1d_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        trend_up = close[i] > ema50_1d_aligned[i]
        
        if position == 0:
            # Enter long: 1d trend up + price breaks above R1 + volume confirmation
            if trend_up and close[i] > camarilla_r1_1d_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: 1d trend down + price breaks below S1 + volume confirmation
            elif not trend_up and close[i] < camarilla_s1_1d_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: 1d trend turns down or price breaks below S1
            if not trend_up or close[i] < camarilla_s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: 1d trend turns up or price breaks above R1
            if trend_up or close[i] > camarilla_r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
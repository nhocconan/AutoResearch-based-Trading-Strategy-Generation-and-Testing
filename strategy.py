#!/usr/bin/env python3
# 12h_Camarilla_Pivot_R3_S3_Breakout_WeeklyTrend_Volume
# Hypothesis: Uses weekly trend filter with daily Camarilla R3/S3 breakout on 12h timeframe.
# Long when weekly trend up and price breaks above R3 with volume confirmation.
# Short when weekly trend down and price breaks below S3 with volume confirmation.
# Weekly trend determined by price above/below weekly EMA34.
# Weekly trend filter reduces whipsaw in ranging markets and improves performance in both bull and bear cycles.
# Target: 15-30 trades/year per symbol with disciplined risk management.

name = "12h_Camarilla_Pivot_R3_S3_Breakout_WeeklyTrend_Volume"
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 34:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly EMA34 for trend filter
    ema34_weekly = np.full_like(close_weekly, np.nan)
    if len(close_weekly) >= 34:
        ema34_weekly[33] = np.mean(close_weekly[0:34])
        for i in range(34, len(close_weekly)):
            ema34_weekly[i] = (close_weekly[i] * 2 + ema34_weekly[i-1] * 32) / 34
    
    ema34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Get daily data for Camarilla levels
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate Camarilla levels (R3, S3)
    camarilla_R3 = np.full_like(close_daily, np.nan)
    camarilla_S3 = np.full_like(close_daily, np.nan)
    
    for i in range(len(df_daily)):
        if i == 0:
            continue
        high_prev = high_daily[i-1]
        low_prev = low_daily[i-1]
        close_prev = close_daily[i-1]
        range_val = high_prev - low_prev
        camarilla_R3[i] = close_prev + range_val * 1.1 / 4
        camarilla_S3[i] = close_prev - range_val * 1.1 / 4
    
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_S3)
    
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
    
    start_idx = max(34, 20, 1)  # Need weekly EMA, daily Camarilla, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_weekly_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        weekly_up = close[i] > ema34_weekly_aligned[i]
        
        if position == 0:
            # Enter long: weekly trend up + price breaks above R3 + volume confirmation
            if weekly_up and close[i] > camarilla_R3_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly trend down + price breaks below S3 + volume confirmation
            elif not weekly_up and close[i] < camarilla_S3_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly trend turns down or price breaks below S3
            if not weekly_up or close[i] < camarilla_S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly trend turns up or price breaks above R3
            if weekly_up or close[i] > camarilla_R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
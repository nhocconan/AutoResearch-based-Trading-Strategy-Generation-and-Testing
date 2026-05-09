#!/usr/bin/env python3
# 1d_WklyTrend_DlyPullback_WithVol
# Hypothesis: Buy weekly trend pullbacks on daily timeframe with volume confirmation.
# Long when weekly trend is up, price pulls back to daily EMA21 with volume > 1.3x average.
# Short when weekly trend is down, price bounces to daily EMA21 with volume > 1.3x average.
# Uses weekly trend filter to avoid counter-trend trades, daily EMA for entry timing,
# and volume spike to confirm institutional interest. Designed for low trade frequency.

name = "1d_WklyTrend_DlyPullback_WithVol"
timeframe = "1d"
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
    if len(df_1w) < 5:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34 for trend filter
    ema34_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 34:
        ema34_1w[33] = np.mean(close_1w[0:34])
        for i in range(34, len(close_1w)):
            ema34_1w[i] = (close_1w[i] * 2 + ema34_1w[i-1] * 32) / 34
    
    # Align weekly EMA to daily timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate daily EMA21 for entry timing
    ema21_daily = np.full_like(close, np.nan)
    if len(close) >= 21:
        ema21_daily[20] = np.mean(close[0:21])
        for i in range(21, len(close)):
            ema21_daily[i] = (close[i] * 2 + ema21_daily[i-1] * 19) / 21
    
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
    
    start_idx = max(34, 21, 20)  # Need weekly EMA, daily EMA, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(ema21_daily[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        weekly_up = close[i] > ema34_1w_aligned[i]
        
        if position == 0:
            # Enter long: weekly trend up + pullback to EMA21 + volume confirmation
            if weekly_up and close[i] <= ema21_daily[i] * 1.005 and volume_ratio[i] > 1.3:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly trend down + bounce to EMA21 + volume confirmation
            elif not weekly_up and close[i] >= ema21_daily[i] * 0.995 and volume_ratio[i] > 1.3:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly trend turns down or price moves above EMA21
            if not weekly_up or close[i] > ema21_daily[i] * 1.01:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly trend turns up or price moves below EMA21
            if weekly_up or close[i] < ema21_daily[i] * 0.99:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
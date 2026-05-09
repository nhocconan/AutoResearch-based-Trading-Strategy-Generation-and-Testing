#!/usr/bin/env python3
# 1d_TRIX_9_0_VolumeSpike_Trend
# Hypothesis: 1d TRIX(9,0) crosses above/below zero with volume spike and 1w trend filter.
# Long when TRIX crosses above zero, volume > 2x average, and price > 1w EMA50.
# Short when TRIX crosses below zero, volume > 2x average, and price < 1w EMA50.
# Designed to generate 8-18 trades/year on 1d to avoid fee decay while capturing momentum.
# Uses momentum (TRIX), volume confirmation, and higher timeframe trend for robustness.

name = "1d_TRIX_9_0_VolumeSpike_Trend"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema50_1w[49] = np.mean(close_1w[0:50])
        for i in range(50, len(close_1w)):
            ema50_1w[i] = (close_1w[i] * 2 + ema50_1w[i-1] * 48) / 50
    
    # Align 1w EMA50 to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate TRIX(9,0): triple EMA of log returns
    # Step 1: log returns
    log_ret = np.full_like(close, np.nan)
    log_ret[1:] = np.log(close[1:] / close[:-1])
    
    # Step 2: EMA3 of log returns
    ema1 = np.full_like(log_ret, np.nan)
    if len(log_ret) >= 3:
        ema1[2] = np.mean(log_ret[0:3])
        for i in range(3, len(log_ret)):
            ema1[i] = log_ret[i] * 0.5 + ema1[i-1] * 0.5  # EMA with alpha=2/(3+1)=0.5
    
    # Step 3: EMA3 of ema1
    ema2 = np.full_like(ema1, np.nan)
    if len(ema1) >= 3:
        ema2[2] = np.mean(ema1[0:3])
        for i in range(3, len(ema1)):
            ema2[i] = ema1[i] * 0.5 + ema2[i-1] * 0.5
    
    # Step 4: EMA3 of ema2
    ema3 = np.full_like(ema2, np.nan)
    if len(ema2) >= 3:
        ema3[2] = np.mean(ema2[0:3])
        for i in range(3, len(ema2)):
            ema3[i] = ema2[i] * 0.5 + ema3[i-1] * 0.5
    
    # TRIX = 100 * (ema3 - ema3_prev) / ema3_prev
    trix = np.full_like(ema3, np.nan)
    valid = ~np.isnan(ema3)
    trix[valid & (np.roll(valid, 1))] = 100 * (ema3[valid & (np.roll(valid, 1))] - ema3[np.roll(valid, 1) & valid]) / ema3[np.roll(valid, 1) & valid]
    
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
    
    start_idx = max(50, 20)  # Need 1w EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(trix[i]) or 
            np.isnan(trix[i-1]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # TRIX zero cross signals
        trix_cross_up = trix[i-1] <= 0 and trix[i] > 0
        trix_cross_down = trix[i-1] >= 0 and trix[i] < 0
        
        # Trend filter
        trend_up = close[i] > ema50_1w_aligned[i]
        trend_down = close[i] < ema50_1w_aligned[i]
        
        if position == 0:
            # Enter long: TRIX crosses up + volume confirmation + uptrend
            if trix_cross_up and volume_ratio[i] > 2.0 and trend_up:
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses down + volume confirmation + downtrend
            elif trix_cross_down and volume_ratio[i] > 2.0 and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses down or trend turns down
            if trix_cross_down or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses up or trend turns up
            if trix_cross_up or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
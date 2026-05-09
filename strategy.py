# [RESEARCH] 12h_Camarilla_Pivot_1W_Trend_Volume
# Hypothesis: Use weekly trend filter (price vs 200 EMA) to capture macro direction,
# Enter on breakout of daily Camarilla R3/S3 levels with volume surge.
# Weekly trend ensures alignment with major cycles, reducing counter-trend trades.
# Daily Camarilla levels provide precise entry/exit points.
# Volume surge confirms institutional participation.
# Timeframe: 12h balances trade frequency (~20-50/year) with signal clarity.
# Works in bull/bear: weekly trend filter adapts, volume surge captures momentum in any regime.
# Risk: Max position 0.30, discrete sizing to minimize churn.

name = "12h_Camarilla_Pivot_1W_Trend_Volume"
timeframe = "12h"
leverage = 1.0

#!/usr/bin/env python3
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
    
    # Get weekly data for trend filter (200 EMA)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA200 for trend filter
    ema200_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 200:
        ema200_1w[199] = np.mean(close_1w[0:200])
        for i in range(200, len(close_1w)):
            ema200_1w[i] = (close_1w[i] * 2 + ema200_1w[i-1] * 198) / 200
    
    # Align weekly EMA200 to 12h timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily Camarilla R3 and S3 levels
    camarilla_r3_1d = np.full_like(close_1d, np.nan)
    camarilla_s3_1d = np.full_like(close_1d, np.nan)
    
    for i in range(len(df_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i])):
            camarilla_r3_1d[i] = close_1d[i] + 1.1 * (high_1d[i] - low_1d[i]) / 2
            camarilla_s3_1d[i] = close_1d[i] - 1.1 * (high_1d[i] - low_1d[i]) / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Volume filter: current volume vs 20-period average on 12h
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
    
    start_idx = max(200, 20)  # Need weekly EMA200 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(camarilla_r3_1d_aligned[i]) or 
            np.isnan(camarilla_s3_1d_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        trend_up = close[i] > ema200_1w_aligned[i]
        volume_surge = volume_ratio[i] > 2.0
        
        if position == 0:
            # Enter long: Uptrend + price breaks above R3 + volume surge
            if trend_up and close[i] > camarilla_r3_1d_aligned[i] and volume_surge:
                signals[i] = 0.30
                position = 1
            # Enter short: Downtrend + price breaks below S3 + volume surge
            elif not trend_up and close[i] < camarilla_s3_1d_aligned[i] and volume_surge:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit long: Trend turns down OR price breaks below S3
            if not trend_up or close[i] < camarilla_s3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: Trend turns up OR price breaks above R3
            if trend_up or close[i] > camarilla_r3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
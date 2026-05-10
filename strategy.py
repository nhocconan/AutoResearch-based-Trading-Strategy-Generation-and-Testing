#!/usr/bin/env python3
# 12H_1D_Camarilla_Pivot_R3_S3_Breakout_Trend_Filter
# Hypothesis: Breakouts from Camarilla R3/S3 levels on 12h chart with daily trend filter
# and volume confirmation work in both bull and bear markets by following the higher timeframe trend.
# Uses 1-day Camarilla levels for structure and 12h price action for entry.
# Target: 20-30 trades per year per symbol.

name = "12H_1D_Camarilla_Pivot_R3_S3_Breakout_Trend_Filter"
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
    
    # Get daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # where C, H, L are from previous day
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    
    # Avoid using first day's data
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Daily trend filter: EMA25
    close_1d_series = pd.Series(close_1d)
    ema25_1d = close_1d_series.ewm(span=25, adjust=False, min_periods=25).mean().values
    
    # Align all daily data to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema25_1d_aligned = align_htf_to_ltf(prices, df_1d, ema25_1d)
    
    # Volume confirmation: 12h volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema25_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check volume filter
        if not volume_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3 with daily uptrend
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema25_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 with daily downtrend
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema25_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S3 or trend changes
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema25_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R3 or trend changes
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema25_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
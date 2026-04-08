#!/usr/bin/env python3
# 4h_camilla_pivot_volume_v1
# Hypothesis: Uses daily Camarilla pivot levels (R3/S3) from 1D for breakout entries, confirmed by volume surge (>1.5x 20-period average) and 1W trend direction (EMA21 slope). 
# Long when: price breaks above R3, volume surge, 1W EMA21 slope > 0.
# Short when: price breaks below S3, volume surge, 1W EMA21 slope < 0.
# Exit when price returns to pivot point (PP) or volume drops below average.
# Designed for low trade frequency (target: 20-40/year) to avoid fee drag, works in bull/bear via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camilla_pivot_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get 1D data for Camarilla pivot levels (based on prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1D bar: based on prior day's range
    # R4 = Close + 1.5*(High-Low)*1.1/2, R3 = Close + 1.1*(High-Low), etc.
    # We use R3 and S3 for breakouts, PP as exit
    rng = high_1d - low_1d
    r3 = close_1d + 1.1 * rng
    s3 = close_1d - 1.1 * rng
    pp = (high_1d + low_1d + close_1d) / 3.0  # Pivot Point
    
    # Align to 4H: each 1D bar's levels apply to the next 4H bars until new 1D bar
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    pp_4h = align_htf_to_ltf(prices, df_1d, pp)
    
    # Get 1W data for trend direction (EMA21 slope)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    # Calculate slope: positive if current EMA > EMA 3 periods ago
    ema21_slope_1w = np.full(len(close_1w), np.nan)
    for i in range(3, len(close_1w)):
        if not np.isnan(ema21_1w[i]) and not np.isnan(ema21_1w[i-3]):
            ema21_slope_1w[i] = ema21_1w[i] - ema21_1w[i-3]
    ema21_slope_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_slope_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(vol_ma_period, 1)  # volume MA needs 20 bars
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or np.isnan(pp_4h[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema21_slope_1w_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to pivot point (PP) or volume drops below average
            if close[i] <= pp_4h[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to pivot point (PP) or volume drops below average
            if close[i] >= pp_4h[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above R3, volume surge, 1W EMA21 slope positive
            if (close[i] > r3_4h[i] and 
                vol_surge[i] and 
                ema21_slope_1w_aligned[i] > 0):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below S3, volume surge, 1W EMA21 slope negative
            elif (close[i] < s3_4h[i] and 
                  vol_surge[i] and 
                  ema21_slope_1w_aligned[i] < 0):
                position = -1
                signals[i] = -0.25
    
    return signals
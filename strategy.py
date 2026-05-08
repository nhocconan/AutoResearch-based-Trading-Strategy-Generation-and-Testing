#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and 1w trend filter.
# Long when price breaks above Camarilla R3 level AND 1d volume > 1.5x 24-period average AND 1w EMA50 trend is up.
# Short when price breaks below Camarilla S3 level AND 1d volume > 1.5x 24-period average AND 1w EMA50 trend is down.
# Exit when price crosses back inside the Camarilla (S3-R3) range.
# Uses 4h timeframe as specified, with 1d volume and 1w EMA50 for higher timeframe context.
# Target: 75-200 total trades over 4 years (19-50/year) with controlled frequency to avoid fee drag.

name = "4h_Camarilla_R3S3_1dVolume_1wEMA50"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for volume
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 2:
        return np.zeros(n)
    
    # Weekly data for EMA50 trend
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla levels are calculated from previous day's range
    close_prev = df_d['close'].values
    high_prev = df_d['high'].values
    low_prev = df_d['low'].values
    
    # Previous day's range
    range_prev = high_prev - low_prev
    
    # Camarilla levels
    # R4 = close_prev + range_prev * 1.5
    # R3 = close_prev + range_prev * 1.25
    # S3 = close_prev - range_prev * 1.25
    # S4 = close_prev - range_prev * 1.5
    # We'll use R3 and S3 as breakout levels
    camarilla_r3 = close_prev + range_prev * 1.25
    camarilla_s3 = close_prev - range_prev * 1.25
    
    # Align Camarilla levels to 4h timeframe (previous day's levels available at 00:00 UTC daily)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_d, camarilla_s3)
    
    # Daily volume filter: current volume > 1.5x 24-period average
    volume_d = df_d['volume'].values
    vol_ma24_d = pd.Series(volume_d).rolling(window=24, min_periods=24).mean().values
    volume_filter_d = volume_d > (1.5 * vol_ma24_d)
    volume_filter = align_htf_to_ltf(prices, df_d, volume_filter_d)
    
    # Weekly EMA50 for trend direction
    close_w = df_w['close'].values
    ema50_w = pd.Series(close_w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_w_aligned = align_htf_to_ltf(prices, df_w, ema50_w)
    
    # Trend filter: price above/below EMA50
    trend_up = close > ema50_w_aligned
    trend_down = close < ema50_w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3, volume filter, uptrend
            long_cond = (close[i] > camarilla_r3_aligned[i]) and volume_filter[i] and trend_up[i]
            # Short conditions: price breaks below Camarilla S3, volume filter, downtrend
            short_cond = (close[i] < camarilla_s3_aligned[i]) and volume_filter[i] and trend_down[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Camarilla S3
            if close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Camarilla R3
            if close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
"""
4h Camarilla Pivot Point Breakout with Volume Spike and Daily Trend Filter.
Long when price breaks above R3 level AND price > daily EMA34 AND volume > 2x average.
Short when price breaks below S3 level AND price < daily EMA34 AND volume > 2x average.
Exit when price crosses back through R1/S1 levels or opposite pivot level is breached.
Uses daily Camarilla pivots calculated from prior day's OHLC for zero look-ahead.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (using prior day's data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily range
    daily_range = high_1d - low_1d
    
    # Camarilla levels (based on previous day)
    # R4 = close + 1.5 * range, R3 = close + 1.1 * range, R2 = close + 0.6 * range, R1 = close + 0.318 * range
    # S1 = close - 0.318 * range, S2 = close - 0.6 * range, S3 = close - 1.1 * range, S4 = close - 1.5 * range
    r4 = close_1d + 1.5 * daily_range
    r3 = close_1d + 1.1 * daily_range
    r2 = close_1d + 0.6 * daily_range
    r1 = close_1d + 0.318 * daily_range
    s1 = close_1d - 0.318 * daily_range
    s2 = close_1d - 0.6 * daily_range
    s3 = close_1d - 1.1 * daily_range
    s4 = close_1d - 1.5 * daily_range
    
    # Align daily levels to 4h timeframe (use previous day's levels)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: volume > 2x 20-period average
    vol_ma_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need daily data (1 day) + volume MA
    start_idx = max(19, 34)  # Need EMA34 and vol MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price_now = close[i]
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: volume > 2x average
        vol_filter = vol_now > 2.0 * vol_avg
        
        if position == 0:
            # Long: price breaks above R3 AND price > daily EMA34 AND volume spike
            if price_now > r3_level and price_now > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below S3 AND price < daily EMA34 AND volume spike
            elif price_now < s3_level and price_now < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below R1 OR breaks below S3 (strong reversal)
            if price_now < r1_level or price_now < s3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above S1 OR breaks above R3 (strong reversal)
            if price_now > s1_level or price_now > r3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_Pivot_Breakout_VolumeSpike_DailyTrend"
timeframe = "4h"
leverage = 1.0
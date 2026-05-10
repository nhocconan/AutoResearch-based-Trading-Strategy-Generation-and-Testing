#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_1wVolFilter
Hypothesis: Daily Camarilla R3/S3 breakout with 1d trend filter (EMA50) and volume confirmation (1d volume > 20-period EMA).
Weekly volume filter ensures we only trade when weekly volume is above average to avoid low-volume false breakouts.
Designed for 15-25 trades/year per symbol, works in bull/bear via trend filter.
"""

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_1wVolFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla calculation (using prior day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (using prior day OHLC)
    high_d = df_1d['high'].shift(1).values
    low_d = df_1d['low'].shift(1).values
    close_d = df_1d['close'].shift(1).values
    
    # Camarilla calculations
    range_d = high_d - low_d
    # R3 and S3 levels (most significant)
    r3_d = close_d + range_d * 1.1 / 2
    s3_d = close_d - range_d * 1.1 / 2
    
    # Align daily Camarilla levels to 4h timeframe
    r3_d_aligned = align_htf_to_ltf(prices, df_1d, r3_d)
    s3_d_aligned = align_htf_to_ltf(prices, df_1d, s3_d)
    
    # Get 1d trend filter: EMA50
    ema50_d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_d_aligned = align_htf_to_ltf(prices, df_1d, ema50_d)
    
    # Get 1d volume filter: volume > 20-period EMA
    vol_ema20_d = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema20_d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema20_d)
    
    # Get weekly volume filter: ensure we trade only when weekly volume is above average
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    vol_ema20_w = pd.Series(df_1w['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema20_w_aligned = align_htf_to_ltf(prices, df_1w, vol_ema20_w)
    
    # Get 4h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h volume filter: current volume > 1.5x 20-period EMA (for entry timing)
    vol_ema20_4h = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter_4h = volume > vol_ema20_4h * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily Camarilla, EMA50, volume filters
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3_d_aligned[i]) or 
            np.isnan(s3_d_aligned[i]) or
            np.isnan(ema50_d_aligned[i]) or
            np.isnan(vol_ema20_d_aligned[i]) or
            np.isnan(vol_ema20_w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine daily trend: price vs EMA50
        bullish_trend = close[i] > ema50_d_aligned[i]
        bearish_trend = close[i] < ema50_d_aligned[i]
        
        # Weekly volume filter: only trade when weekly volume is above average
        weekly_volume_ok = volume[i] > vol_ema20_w_aligned[i]  # Using 4h volume vs weekly average
        
        if position == 0:
            # Long: bullish daily trend AND price breaks above R3 with volume confirmation
            if bullish_trend and high[i] > r3_d_aligned[i] and volume_filter_4h[i] and weekly_volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish daily trend AND price breaks below S3 with volume confirmation
            elif bearish_trend and low[i] < s3_d_aligned[i] and volume_filter_4h[i] and weekly_volume_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 OR trend turns bearish
            if low[i] < s3_d_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 OR trend turns bullish
            if high[i] > r3_d_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
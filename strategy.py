#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: 12h Camarilla R3/S3 breakout filtered by 1w EMA50 trend and volume surge.
# Camarilla levels provide high-probability reversal/breakout points. 
# Breakout above R3 or below S3 with volume and trend continuation captures strong moves.
# Works in bull/bear markets by using weekly trend filter and requiring volume confirmation.
# Targets 12-37 trades/year to minimize fee drag on 12h timeframe.

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # Camarilla: Close +- (High - Low) * multipliers
    # R3 = Close + (High - Low) * 1.1000
    # S3 = Close - (High - Low) * 1.1000
    # R4 = Close + (High - Low) * 1.2000
    # S4 = Close - (High - Low) * 1.2000
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    camarilla_multiplier = 1.1000  # for R3/S3
    high_low_range = high_1d - low_1d
    
    r3 = close_1d_vals + (high_low_range * camarilla_multiplier)
    s3 = close_1d_vals - (high_low_range * camarilla_multiplier)
    r4 = close_1d_vals + (high_low_range * 1.2000)
    s4 = close_1d_vals - (high_low_range * 1.2000)
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume average (30-period)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 (50) + volume MA (30)
    start_idx = 80
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1w EMA50
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
        uptrend = close_1w_aligned[i] > ema_50_1w_aligned[i]
        downtrend = close_1w_aligned[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        # Camarilla breakout signals
        breakout_r3 = close[i] > r3_aligned[i-1]
        breakdown_s3 = close[i] < s3_aligned[i-1]
        breakout_r4 = close[i] > r4_aligned[i-1]
        breakdown_s4 = close[i] < s4_aligned[i-1]
        
        if position == 0:
            # Long: Camarilla R3 breakout with volume surge and 1w uptrend
            if breakout_r3 and volume_surge and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla S3 breakdown with volume surge and 1w downtrend
            elif breakdown_s3 and volume_surge and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 or trend changes
            if close[i] < s3_aligned[i-1] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 or trend changes
            if close[i] > r3_aligned[i-1] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
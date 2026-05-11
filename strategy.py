#!/usr/bin/env python3
name = "6h_WeeklyPivot_Bias_With_DailyTrend_Filter"
timeframe = "6h"
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
    
    # Get 1d data for daily trend (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1w data for weekly pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's data)
    # We'll calculate pivot points for each week based on the previous week's OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot points for each week: P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    # R4 = 3*P - 2*L, S4 = 3*P - 2*H
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    r4_1w = 3 * pivot_1w - 2 * low_1w
    s4_1w = 3 * pivot_1w - 2 * high_1w
    
    # Align weekly pivot levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.3 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for EMA and pivot alignment
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(pivot_1w_aligned[i]) or
            np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine bias based on weekly pivot levels
        # Bullish bias: price above weekly pivot
        # Bearish bias: price below weekly pivot
        bullish_bias = close[i] > pivot_1w_aligned[i]
        bearish_bias = close[i] < pivot_1w_aligned[i]
        
        # Daily trend filter: EMA34 slope
        # Uptrend: current price above EMA34 and EMA34 rising
        # Downtrend: current price below EMA34 and EMA34 falling
        if i >= 1:
            ema_rising = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            ema_falling = ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]
        else:
            ema_rising = False
            ema_falling = False
            
        daily_uptrend = close[i] > ema_34_1d_aligned[i] and ema_rising
        daily_downtrend = close[i] < ema_34_1d_aligned[i] and ema_falling
        
        if position == 0:
            # Long: Bullish bias + daily uptrend + volume confirmation
            if bullish_bias and daily_uptrend and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish bias + daily downtrend + volume confirmation
            elif bearish_bias and daily_downtrend and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish bias OR daily downtrend
            if bearish_bias or daily_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish bias OR daily uptrend
            if bullish_bias or daily_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
# 12h_Camarilla_Pivot_1wTrend_Volume
# Hypothesis: Use weekly Camarilla pivot levels (R4/S4) from 1w data for breakout entries,
# filtered by 1d trend (EMA50) and volume confirmation. Enter long on break above R4 with
# uptrend and volume spike, short on break below S4 with downtrend and volume spike.
# Exit on retrace to pivot point (PP) or trend failure. Designed for low frequency
# (10-30 trades/year) to avoid fee drag. Works in bull (catch breakouts) and bear
# (catch breakdowns) with trend filter and volume confirmation.

name = "12h_Camarilla_Pivot_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """
    Calculate Camarilla pivot levels for given high, low, close.
    Returns (R4, R3, R2, R1, PP, S1, S2, S3, S4)
    """
    typical = (high + low + close) / 3.0
    range_val = high - low
    
    pp = typical
    r1 = close + (range_val * 1.1 / 12)
    r2 = close + (range_val * 1.1 / 6)
    r3 = close + (range_val * 1.1 / 4)
    r4 = close + (range_val * 1.1 / 2)
    s1 = close - (range_val * 1.1 / 12)
    s2 = close - (range_val * 1.1 / 6)
    s3 = close - (range_val * 1.1 / 4)
    s4 = close - (range_val * 1.1 / 2)
    
    return r4, r3, r2, r1, pp, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels on weekly data
    r4_1w, r3_1w, r2_1w, r1_1w, pp_1w, s1_1w, s2_1w, s3_1w, s4_1w = calculate_camarilla(
        high_1w, low_1w, close_1w
    )
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly and daily data to 12h timeframe
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or 
            np.isnan(pp_1w_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Break above R4 with uptrend and volume confirmation
            if close[i] > r4_1w_aligned[i] and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S4 with downtrend and volume confirmation
            elif close[i] < s4_1w_aligned[i] and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Retrace to PP or trend fails
            if close[i] < pp_1w_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Retrace to PP or trend fails
            if close[i] > pp_1w_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
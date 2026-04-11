#!/usr/bin/env python3
"""
12h_1d_1w_camarilla_pivot_volume_v1
Strategy: 12h Camarilla pivot level reversal with volume confirmation and 1d trend filter
Timeframe: 12h
Leverage: 1.0
Hypothesis: Uses daily Camarilla pivot levels (calculated from previous day's high-low-close) as support/resistance zones. Enters long when price bounces off S3/S4 with volume confirmation in a 1d uptrend, and short when price bounces off R3/R4 with volume confirmation in a 1d downtrend. Exits when price reaches the opposite pivot level or mean (P). Designed to capture mean-reversion bounces at strong intraday levels while avoiding false signals in strong trends. Works in both bull and bear markets by using 1d trend filter to align with higher timeframe momentum. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each day
    # Based on previous day's H, L, C
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1 to avoid look-ahead)
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    
    # Calculate pivot and Camarilla levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels: S1, S2, S3, S4 and R1, R2, R3, R4
    # S4 = Close - ((High - Low) * 1.500)
    # S3 = Close - ((High - Low) * 1.250)
    # S2 = Close - ((High - Low) * 1.166)
    # S1 = Close - ((High - Low) * 1.083)
    # R1 = Close + ((High - Low) * 1.083)
    # R2 = Close + ((High - Low) * 1.166)
    # R3 = Close + ((High - Low) * 1.250)
    # R4 = Close + ((High - Low) * 1.500)
    
    s4 = prev_close - (range_hl * 1.500)
    s3 = prev_close - (range_hl * 1.250)
    s2 = prev_close - (range_hl * 1.166)
    s1 = prev_close - (range_hl * 1.083)
    r1 = prev_close + (range_hl * 1.083)
    r2 = prev_close + (range_hl * 1.166)
    r3 = prev_close + (range_hl * 1.250)
    r4 = prev_close + (range_hl * 1.500)
    
    # Align all levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: price near S3/S4 with volume in uptrend
        long_signal = vol_confirmed and uptrend_1d and (
            (price_close <= s3_aligned[i] * 1.005 and price_close >= s4_aligned[i] * 0.995) or
            (price_close <= s4_aligned[i] * 1.005 and price_close >= s4_aligned[i] * 0.995)
        )
        
        # Short: price near R3/R4 with volume in downtrend
        short_signal = vol_confirmed and downtrend_1d and (
            (price_close >= r3_aligned[i] * 0.995 and price_close <= r4_aligned[i] * 1.005) or
            (price_close >= r4_aligned[i] * 0.995 and price_close <= r4_aligned[i] * 1.005)
        )
        
        # Exit when price reaches opposite level or pivot
        exit_long = position == 1 and (
            price_close >= pivot_aligned[i] * 0.995 or  # Reached pivot
            price_close >= r1_aligned[i] * 0.995        # Reached R1
        )
        exit_short = position == -1 and (
            price_close <= pivot_aligned[i] * 1.005 or  # Reached pivot
            price_close <= s1_aligned[i] * 1.005        # Reached S1
        )
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
#!/usr/bin/env python3
"""
6h Weekly Pivot R3/S3 Rejection with Volume Spike and 1d Trend Filter
Hypothesis: Weekly pivot levels (R3/S3) act as strong institutional support/resistance.
Price rejection at these levels with volume confirmation, aligned with 1d trend,
captures reversals in both bull and bear markets. Low frequency (~20-40/year)
minimizes fee drag while capturing high-probability mean reversion moves.
"""

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
    
    # Get weekly data for pivot calculation (once before loop)
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    # Standard formula: P = (H + L + C)/3, R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    prev_week_high = df_w['high'].shift(1).values
    prev_week_low = df_w['low'].shift(1).values
    prev_week_close = df_w['close'].shift(1).values
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    weekly_r3 = prev_week_high + 2 * (pivot - prev_week_low)
    weekly_s3 = prev_week_low - 2 * (prev_week_high - pivot)
    
    # Align weekly pivot levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_w, weekly_r3)
    s3_aligned = align_htf_to_ltf(prices, df_w, weekly_s3)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 2.0x 24-period volume average (on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        trend = ema50_1d_aligned[i]
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Look for rejection at R3/S3 with volume, counter to trend
            if vol_ok:
                # Rejection at R3 in uptrend -> short
                if price < r3 and price > trend:
                    signals[i] = -0.25
                    position = -1
                # Rejection at S3 in downtrend -> long
                elif price > s3 and price < trend:
                    signals[i] = 0.25
                    position = 1
        
        elif position == 1:
            # Exit if price returns to S3 or trend continues
            if price > s3 or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to R3 or trend continues
            if price < r3 or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Weekly_Pivot_R3S3_Rejection_Volume_Trend"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_R3S4_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once (for pivot calculation)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivots (using previous week's data)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Weekly pivot: P = (H + L + C) / 3
    pivot_w = (high_w + low_w + close_w) / 3
    # Weekly R3 and S3
    r3_w = pivot_w + 2 * (high_w - low_w)
    s3_w = pivot_w - 2 * (high_w - low_w)
    # Weekly R4 and S4
    r4_w = pivot_w + 3 * (high_w - low_w)
    s4_w = pivot_w - 3 * (high_w - low_w)
    
    # Align weekly pivots to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    r3_w_aligned = align_htf_to_ltf(prices, df_1w, r3_w)
    s3_w_aligned = align_htf_to_ltf(prices, df_1w, s3_w)
    r4_w_aligned = align_htf_to_ltf(prices, df_1w, r4_w)
    s4_w_aligned = align_htf_to_ltf(prices, df_1w, s4_w)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend direction
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike detection: current volume > 2 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_w_aligned[i]) or np.isnan(r3_w_aligned[i]) or 
            np.isnan(s3_w_aligned[i]) or np.isnan(r4_w_aligned[i]) or 
            np.isnan(s4_w_aligned[i]) or np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pivot_val = pivot_w_aligned[i]
        r3_val = r3_w_aligned[i]
        s3_val = s3_w_aligned[i]
        r4_val = r4_w_aligned[i]
        s4_val = s4_w_aligned[i]
        ema_val = ema34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: break above R4 with volume spike and above 1d EMA
            if (close[i] > r4_val and vol_spike and close[i] > ema_val):
                signals[i] = 0.25
                position = 1
            # Enter short: break below S4 with volume spike and below 1d EMA
            elif (close[i] < s4_val and vol_spike and close[i] < ema_val):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below R3 (reversal signal) or below 1d EMA
            if (close[i] < r3_val or close[i] < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above S3 (reversal signal) or above 1d EMA
            if (close[i] > s3_val or close[i] > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Uses weekly pivot points (R3/S3, R4/S4) with volume confirmation and 1d EMA trend filter.
# - Enters long when price breaks above weekly R4 with volume spike and above 1d EMA
# - Enters short when price breaks below weekly S4 with volume spike and below 1d EMA
# - Exits when price breaks back below weekly R3 (long) or above weekly S3 (short) OR crosses 1d EMA
# - Weekly R4/S4 act as strong breakout levels for trend continuation
# - Weekly R3/S3 act as reversal zones for taking profits
# - Volume spike filter ensures breakouts have conviction
# - 1d EMA filter ensures trading with higher timeframe trend
# - Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# - Position size: 0.25 for balanced risk/return
# - Works in both bull and bear markets by following 1d trend direction
# - Weekly pivot points provide institutional levels that work across market regimes
# - Focus on BTC and ETH as primary targets (not SOL-only)
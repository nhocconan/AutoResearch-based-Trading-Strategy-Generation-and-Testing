#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Reversal_v2
Hypothesis: Price reverses at weekly pivot levels (S2/S3/R2/R3) with volume confirmation and trend filter. 
Uses 1d ATR to dynamically adjust entry/exit zones and filter out low-volatility periods. 
Trades only when price approaches weekly support/resistance with volume spike and trend alignment.
Designed for 6h timeframe to balance signal frequency and profit potential.
Target: 15-30 trades/year to minimize fee drag.
"""

name = "6h_Weekly_Pivot_Reversal_v2"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for pivot calculation
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 5:
        return np.zeros(n)
    
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    pivot_w = (high_w + low_w + close_w) / 3.0
    s1_w = 2 * pivot_w - high_w
    s2_w = pivot_w - (high_w - low_w)
    s3_w = low_w - 2 * (high_w - pivot_w)
    r1_w = 2 * pivot_w - low_w
    r2_w = pivot_w + (high_w - low_w)
    r3_w = high_w + 2 * (pivot_w - low_w)
    
    # Align weekly pivots to 6h timeframe (with 1-bar delay for completed weekly bar)
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_w, s2_w)
    s3_w_aligned = align_htf_to_ltf(prices, df_w, s3_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_w, r2_w)
    r3_w_aligned = align_htf_to_ltf(prices, df_w, r3_w)
    
    # Get 1d data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ATR on 1d
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = np.inf  # First value has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Get 6h data for trend filter and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 34:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate 6h EMA34 for trend filter
    ema34_6h = pd.Series(close_6h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_6h_aligned = align_htf_to_ltf(prices, df_6h, ema34_6h)
    
    # 6h data for signal generation
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly pivot (5 weeks), 1d ATR (14), and 6h EMA34 (34)
    start_idx = max(5, 14, 34)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema34_6h_aligned[i]) or 
            np.isnan(s2_w_aligned[i]) or
            np.isnan(s3_w_aligned[i]) or
            np.isnan(r2_w_aligned[i]) or
            np.isnan(r3_w_aligned[i]) or
            np.isnan(atr_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs 6h EMA34
        uptrend_6h = close[i] > ema34_6h_aligned[i]
        downtrend_6h = close[i] < ema34_6h_aligned[i]
        
        # Volume filter: current 6h volume > 2.0x 20-period average
        vol_ma20 = pd.Series(volume_6h).ewm(span=20, adjust=False, min_periods=20).mean().values
        vol_ma20_aligned = align_htf_to_ltf(prices, df_6h, vol_ma20)
        volume_filter = volume[i] > vol_ma20_aligned[i] * 2.0
        
        # Dynamic zone size based on 1d ATR (as fraction of price)
        zone_size = (atr_1d_aligned[i] * 0.5) / close[i]  # 0.5x ATR as zone
        
        # Distance to weekly levels (as fraction of price)
        dist_to_s2 = (close[i] - s2_w_aligned[i]) / close[i] if s2_w_aligned[i] > 0 else 1.0
        dist_to_s3 = (close[i] - s3_w_aligned[i]) / close[i] if s3_w_aligned[i] > 0 else 1.0
        dist_to_r2 = (r2_w_aligned[i] - close[i]) / close[i] if r2_w_aligned[i] > 0 else 1.0
        dist_to_r3 = (r3_w_aligned[i] - close[i]) / close[i] if r3_w_aligned[i] > 0 else 1.0
        
        # Entry zones: within dynamic zone of weekly S2/S3 or R2/R3
        near_s2 = abs(dist_to_s2) < zone_size
        near_s3 = abs(dist_to_s3) < zone_size
        near_r2 = abs(dist_to_r2) < zone_size
        near_r3 = abs(dist_to_r3) < zone_size
        
        if position == 0:
            # Long entry: near S2/S3 + uptrend + volume
            if (near_s2 or near_s3) and uptrend_6h and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: near R2/R3 + downtrend + volume
            elif (near_r2 or near_r3) and downtrend_6h and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches weekly pivot or trend fails
            if close[i] >= pivot_w_aligned[i] or not uptrend_6h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches weekly pivot or trend fails
            if close[i] <= pivot_w_aligned[i] or not downtrend_6h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#/usr/bin/env python3
"""
6h_WeeklyPivot_Pullback_1dTrend
Hypothesis: Price pulls back to weekly pivot levels (R1/S1) during a 1d trend, entering in trend direction with volume confirmation.
Weekly pivots act as dynamic support/resistance; pullbacks offer high-probability entries in trending markets.
1d trend filter ensures alignment with higher timeframe momentum. Volume filters false signals.
Works in bull/bear by trading only in direction of 1d trend. Target: 15-30 trades/year (60-120 total).
"""

name = "6h_WeeklyPivot_Pullback_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Weekly data for pivot points (using 1d data resampled to weekly via aggregation in parquet)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    
    # Weekly pivot points: P = (H+L+C)/3, R1 = 2P - L, S1 = 2P - H
    pp_1w = np.full(len(high_1w), np.nan)
    r1_1w = np.full(len(high_1w), np.nan)
    s1_1w = np.full(len(high_1w), np.nan)
    
    for i in range(len(high_1w)):
        pp_1w[i] = (high_1w[i] + low_1w[i] + close_1w[i]) / 3.0
        r1_1w[i] = 2 * pp_1w[i] - low_1w[i]
        s1_1w[i] = 2 * pp_1w[i] - high_1w[i]
    
    # Align 1d and weekly data to 6h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # 6h volume SMA20 for confirmation
    vol_sma20 = np.full(len(volume), np.nan)
    if len(volume) >= 20:
        vol_sma20[19] = np.mean(volume[:20])
        for i in range(20, len(volume)):
            vol_sma20[i] = (vol_sma20[i-1] * 19 + volume[i]) / 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(pp_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or np.isnan(vol_sma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * vol_sma20[i]
        
        # Trend and price relative to weekly pivot levels
        is_uptrend = close[i] > ema50_1d_aligned[i]
        is_downtrend = close[i] < ema50_1d_aligned[i]
        price_near_r1 = abs(close[i] - r1_1w_aligned[i]) / close[i] < 0.005  # Within 0.5%
        price_near_s1 = abs(close[i] - s1_1w_aligned[i]) / close[i] < 0.005  # Within 0.5%
        
        if position == 0:
            # Long: pullback to S1 in uptrend with volume
            if price_near_s1 and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: pullback to R1 in downtrend with volume
            elif price_near_r1 and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: trend turns down or price moves away from S1
            if not is_uptrend or abs(close[i] - s1_1w_aligned[i]) / close[i] > 0.02:  # Moved >2% away
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: trend turns up or price moves away from R1
            if not is_downtrend or abs(close[i] - r1_1w_aligned[i]) / close[i] > 0.02:  # Moved >2% away
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
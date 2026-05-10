#!/usr/bin/env python3
"""
1d_Weekly_Pivot_Pullback_1wTrend
Hypothesis: Price pulls back to weekly pivot support/resistance during strong weekly trends.
In bull markets, buy near weekly S1/S2; in bear markets, sell near weekly R1/R2.
Weekly trend filter ensures alignment with higher timeframe momentum.
Pivot levels provide structure; pullbacks offer better risk/reward than breakouts.
Designed for low trade frequency (<25/year) to minimize fee drag in ranging/bear markets.
Works on BTC/ETH via mean reversion within strong trends.
"""

name = "1d_Weekly_Pivot_Pullback_1wTrend"
timeframe = "1d"
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
    
    # Weekly data for pivot points and trend
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points (using prior week's OHLC)
    # P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    pp_1w = np.full(len(high_1w), np.nan)
    r1_1w = np.full(len(high_1w), np.nan)
    s1_1w = np.full(len(high_1w), np.nan)
    r2_1w = np.full(len(high_1w), np.nan)
    s2_1w = np.full(len(high_1w), np.nan)
    
    if len(high_1w) >= 1:
        for i in range(1, len(high_1w)):
            # Use previous week's data to calculate current week's pivots
            ph = high_1w[i-1]
            pl = low_1w[i-1]
            pc = close_1w[i-1]
            
            pp = (ph + pl + pc) / 3.0
            r1 = 2 * pp - pl
            s1 = 2 * pp - ph
            r2 = pp + (ph - pl)
            s2 = pp - (ph - pl)
            
            pp_1w[i] = pp
            r1_1w[i] = r1
            s1_1w[i] = s1
            r2_1w[i] = r2
            s2_1w[i] = s2
    
    # Weekly EMA50 for trend filter
    ema50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema50_1w[49] = np.mean(close_1w[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema50_1w[i-1]
    
    # Align weekly indicators to daily
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50
    
    for i in range(start_idx, n):
        if np.isnan(pp_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or \
           np.isnan(r2_1w_aligned[i]) or np.isnan(s2_1w_aligned[i]) or np.isnan(ema50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend determination
        is_uptrend = close[i] > ema50_1w_aligned[i]
        is_downtrend = close[i] < ema50_1w_aligned[i]
        
        # Price proximity to pivot levels (within 0.5% for entry)
        proximity_threshold = 0.005  # 0.5%
        near_s1 = abs(close[i] - s1_1w_aligned[i]) / close[i] <= proximity_threshold
        near_s2 = abs(close[i] - s2_1w_aligned[i]) / close[i] <= proximity_threshold
        near_r1 = abs(close[i] - r1_1w_aligned[i]) / close[i] <= proximity_threshold
        near_r2 = abs(close[i] - r2_1w_aligned[i]) / close[i] <= proximity_threshold
        
        if position == 0:
            # Long: pullback to support in uptrend
            if (near_s1 or near_s2) and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: pullback to resistance in downtrend
            elif (near_r1 or near_r2) and is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: trend breaks down or price reaches pivot resistance
            if not is_uptrend or close[i] >= r1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: trend breaks up or price reaches pivot support
            if not is_downtrend or close[i] <= s1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
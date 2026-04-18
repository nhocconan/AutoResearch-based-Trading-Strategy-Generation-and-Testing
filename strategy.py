#!/usr/bin/env python3
"""
6h_Pivot_R3S3_Fade_R4S4_Breakout
Hypothesis: Use daily Camarilla R3/S3 for mean-reversion fades and R4/S4 for breakout continuation.
In ranging markets (identified by Bollinger Bandwidth), fade R3/S3 reversals. In trending markets
(narrow BBW or ADX > 25), break R4/S4 continuation. This adapts to market regime, reducing false
signals in chop and capturing momentum in trends. Designed for 6H timeframe with ~15-35 trades/year.
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
    
    # Get daily data for Camarilla levels and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels (R3/S3 and R4/S4 from prior day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r3 = np.full(len(close_1d), np.nan)  # R3 level
    s3 = np.full(len(close_1d), np.nan)  # S3 level
    r4 = np.full(len(close_1d), np.nan)  # R4 level
    s4 = np.full(len(close_1d), np.nan)  # S4 level
    
    for i in range(1, len(close_1d)):
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        diff = ph - pl
        r3[i] = pc + 1.1 * diff  # R3
        s3[i] = pc - 1.1 * diff  # S3
        r4[i] = pc + 1.618 * diff  # R4
        s4[i] = pc - 1.618 * diff  # S4
    
    # Calculate Bollinger Bandwidth (20,2) for regime detection
    bb_period = 20
    bb_std = 2
    sma = np.full(n, np.nan)
    bb_up = np.full(n, np.nan)
    bb_dn = np.full(n, np.nan)
    bbw = np.full(n, np.nan)
    
    if n >= bb_period:
        for i in range(bb_period - 1, n):
            sma[i] = np.mean(close[i-bb_period+1:i+1])
            std = np.std(close[i-bb_period+1:i+1])
            bb_up[i] = sma[i] + bb_std * std
            bb_dn[i] = sma[i] - bb_std * std
            bbw[i] = (bb_up[i] - bb_dn[i]) / sma[i] if sma[i] != 0 else np.nan
    
    # Align daily levels and BBW to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    bbw_aligned = align_htf_to_ltf(prices, df_1d, bbw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 1)
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(bbw_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime: narrow BBW = trending, wide BBW = ranging
        # Using 50th percentile of BBW as threshold (adaptive)
        if i >= 50:
            bbw_lookback = bbw_aligned[max(0, i-50):i]
            bbw_median = np.nanmedian(bbw_lookback)
            is_trending = bbw_aligned[i] < bbw_median
        else:
            is_trending = False  # default to ranging until enough data
        
        if position == 0:
            if is_trending:
                # Trending: breakout continuation at R4/S4
                if close[i] > r4_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < s4_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                # Ranging: fade at R3/S3
                if close[i] > r3_aligned[i]:
                    signals[i] = -0.25  # short at R3
                    position = -1
                elif close[i] < s3_aligned[i]:
                    signals[i] = 0.25   # long at S3
                    position = 1
        
        elif position == 1:
            # Long exit: in trend - exit on S4 break; in range - exit at midpoint or R3
            if is_trending:
                if close[i] < s4_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                midpoint = (r3_aligned[i] + s3_aligned[i]) / 2
                if close[i] > midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Short exit: in trend - exit on R4 break; in range - exit at midpoint or S3
            if is_trending:
                if close[i] > r4_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                midpoint = (r3_aligned[i] + s3_aligned[i]) / 2
                if close[i] < midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Pivot_R3S3_Fade_R4S4_Breakout"
timeframe = "6h"
leverage = 1.0
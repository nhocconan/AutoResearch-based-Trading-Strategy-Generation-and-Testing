#!/usr/bin/env python3
# 6h_PivotsAndVolumeBreakout
# Hypothesis: Breakouts above/below weekly pivot levels (R4/S4) with volume confirmation and daily trend filter.
# Uses weekly pivot points for structure, daily EMA34 for trend, and volume spike for confirmation.
# Works in both bull and bear markets by trading breakouts in direction of higher timeframe trend.
# Targets ~20-40 trades/year to minimize fee drag.

name = "6h_PivotsAndVolumeBreakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily EMA34 for trend
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align daily trend to 6h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly high, low, close for pivot calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points
    pp = (high_1w + low_1w + close_1w) / 3
    r1 = 2 * pp - low_1w
    s1 = 2 * pp - high_1w
    r2 = pp + (high_1w - low_1w)
    s2 = pp - (high_1w - low_1w)
    r3 = high_1w + 2 * (pp - low_1w)
    s3 = low_1w - 2 * (high_1w - pp)
    r4 = pp + 3 * (high_1w - low_1w)
    s4 = pp - 3 * (high_1w - low_1w)
    
    # Align weekly pivots to 6h
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34
    
    for i in range(start_idx, n):
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R4 with volume spike and daily uptrend
            if (high[i] > r4_aligned[i] and
                vol_spike[i] and
                trend_1d_up_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: break below S4 with volume spike and daily downtrend
            elif (low[i] < s4_aligned[i] and
                  vol_spike[i] and
                  trend_1d_down_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price falls below pivot point or volume dries up
            if (close[i] < pp[-1] if len(pp) > 0 else 0 or
                not vol_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price rises above pivot point or volume dries up
            if (close[i] > pp[-1] if len(pp) > 0 else 0 or
                not vol_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
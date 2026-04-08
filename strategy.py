#!/usr/bin/env python3
# 6h_1d_weekly_pivot_volume_v1
# Hypothesis: Fade at weekly pivot extremes (R4/S4) and breakout continuation beyond R5/S5, with 1d trend filter and volume confirmation.
# Weekly pivots provide institutional reference points; price rejecting S4/R4 suggests mean reversion, while breaking R5/S5 indicates momentum.
# 1d trend filter ensures we only take trades in direction of higher timeframe trend to avoid counter-trend traps.
# Volume confirmation reduces false signals. Works in bull/bear by adapting to weekly structure.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_1d_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly pivot points from 1w data (using 1d resampled internally by get_htf_data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: using prior week's OHLC
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    # R4 = R3 + (H - L), S4 = S3 - (H - L)
    # R5 = R4 + (H - L), S5 = S4 - (H - L)
    
    # We need prior week's data for current week's pivot
    # Shift the weekly OHLC by 1 to use prior week
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_close = df_1w['close'].values
    
    # Use prior week's data to calculate current week's pivot
    wk_high_prev = np.roll(wk_high, 1)
    wk_low_prev = np.roll(wk_low, 1)
    wk_close_prev = np.roll(wk_close, 1)
    # First week has no prior, set to NaN
    wk_high_prev[0] = np.nan
    wk_low_prev[0] = np.nan
    wk_close_prev[0] = np.nan
    
    # Calculate pivot levels using prior week
    pivot = (wk_high_prev + wk_low_prev + wk_close_prev) / 3.0
    rng = wk_high_prev - wk_low_prev
    
    r1 = 2 * pivot - wk_low_prev
    s1 = 2 * pivot - wk_high_prev
    r2 = pivot + rng
    s2 = pivot - rng
    r3 = wk_high_prev + 2 * (pivot - wk_low_prev)
    s3 = wk_low_prev - 2 * (wk_high_prev - pivot)
    r4 = r3 + rng
    s4 = s3 - rng
    r5 = r4 + rng
    s5 = s4 - rng
    
    # Align weekly pivot levels to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    r5_aligned = align_htf_to_ltf(prices, df_1w, r5)
    s5_aligned = align_htf_to_ltf(prices, df_1w, s5)
    
    # 1d trend filter: EMA50 > EMA200 = uptrend, < = downtrend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # 1d trend: 1 if EMA50 > EMA200 (uptrend), -1 if EMA50 < EMA200 (downtrend)
    trend_1d = np.where(ema50_1d > ema200_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Volume spike detection on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_spike = vol_ratio > 1.8  # 80% above average volume
    
    # Session filter: 00-23 UTC (6h bars less sensitive to session, but avoid quiet periods)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 0) & (hours <= 23)  # Always true for 6h, kept for structure
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(r5_aligned[i]) or np.isnan(s5_aligned[i]) or np.isnan(trend_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade with volume spike
        if not vol_spike[i]:
            if position != 0:
                # Hold position until exit signal
                pass
            else:
                signals[i] = 0.0
                continue
        
        if position == 1:  # Long position
            # Exit: price reaches R5 (take profit) or reverses below R4 (stop)
            if close[i] >= r5_aligned[i] or close[i] < r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches S5 (take profit) or reverses above S4 (stop)
            if close[i] <= s5_aligned[i] or close[i] > s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Fade at R4/S4 in ranging markets, breakout beyond R5/S5 in trending markets
            # But we use 1d trend to determine regime
            
            # Long conditions
            long_signal = False
            if trend_1d_aligned[i] == 1:  # Uptrend: look for breakout above R5
                if close[i] > r5_aligned[i]:
                    long_signal = True
            else:  # Downtrend or ranging: look for bounce at S4
                if close[i] < s4_aligned[i] and close[i] > s5_aligned[i]:
                    # Price is between S4 and S5, potential bounce from S4
                    long_signal = True
            
            # Short conditions
            short_signal = False
            if trend_1d_aligned[i] == -1:  # Downtrend: look for breakdown below S5
                if close[i] < s5_aligned[i]:
                    short_signal = True
            else:  # Uptrend or ranging: look for rejection at R4
                if close[i] > r4_aligned[i] and close[i] < r5_aligned[i]:
                    # Price is between R4 and R5, potential rejection from R4
                    short_signal = True
            
            if long_signal:
                position = 1
                signals[i] = 0.25
            elif short_signal:
                position = -1
                signals[i] = -0.25
    
    return signals
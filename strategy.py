#!/usr/bin/env python3
# 6h_donchian_breakout_weekly_pivot_volume_v1
# Hypothesis: Donchian channel breakout on 6h filtered by weekly pivot direction and volume confirmation.
# Works in bull/bear: Pivot provides directional bias, Donchian captures breakouts, volume filters false signals.
# Target: 50-150 total trades over 4 years (~12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Daily pivot points (standard)
    if len(df_1d) >= 1:
        pp = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
        r1 = 2 * pp - df_1d['low']
        s1 = 2 * pp - df_1d['high']
        r2 = pp + (df_1d['high'] - df_1d['low'])
        s2 = pp - (df_1d['high'] - df_1d['low'])
        r3 = df_1d['high'] + 2 * (pp - df_1d['low'])
        s3 = df_1d['low'] - 2 * (df_1d['high'] - pp)
        
        # Align to 6h timeframe (wait for daily close)
        pp_aligned = align_htf_to_ltf(prices, df_1d, pp.values)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    else:
        pp_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
    
    # Weekly trend: price above/below weekly pivot
    if len(df_1w) >= 1:
        wp = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
        wp_aligned = align_htf_to_ltf(prices, df_1w, wp.values)
    else:
        wp_aligned = np.full(n, np.nan)
    
    # Donchian channel (20-period) on 6h
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation (20-period average)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = max(lookback, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or \
           np.isnan(wp_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower or weekly pivot turns bearish
            if close[i] < lower[i] or close[i] < wp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper or weekly pivot turns bullish
            if close[i] > upper[i] or close[i] > wp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long: price breaks above Donchian upper AND price above weekly pivot AND above daily R3 (bullish bias)
            if close[i] > upper[i] and close[i] > wp_aligned[i] and close[i] > r3_aligned[i] and volume_ok:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian lower AND price below weekly pivot AND below daily S3 (bearish bias)
            elif close[i] < lower[i] and close[i] < wp_aligned[i] and close[i] < s3_aligned[i] and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals
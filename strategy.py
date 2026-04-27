#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_Regime
Hypothesis: Daily Camarilla R1/S1 breakout with weekly trend filter (EMA50), volume spike, and choppiness regime.
Only trade breakouts in non-choppy, trending markets on the weekly timeframe.
Designed for lower trade frequency (~10-25/year) to minimize fee drag while capturing strong trends.
Works in bull (breakouts with trend) and bear (avoids false signals in choppy/range markets).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate weekly Camarilla pivot levels (R1, S1) from prior week
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    PP = (high_1w + low_1w + close_1w) / 3.0
    R1 = PP + (high_1w - low_1w) * 1.0 / 4.0
    S1 = PP - (high_1w - low_1w) * 1.0 / 4.0
    
    # Align weekly Camarilla levels to daily timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    
    # Weekly trend filter: EMA50 on weekly close
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily volume spike: current volume > 2.0 * 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # Daily choppiness index (14-period)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.full(n, 50.0)  # default neutral
    for i in range(n):
        if i >= 13 and atr14[i] > 0 and hh14[i] != ll14[i]:
            chop[i] = 100 * np.log10(atr14[i] / (hh14[i] - ll14[i])) / np.log10(14)
    
    # Regime filter: only trade when NOT choppy (CHOP < 61.8 = trending)
    not_choppy = chop < 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for weekly EMA50, volume avg, chop
    start_idx = max(100, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(not_choppy[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for breakout with volume confirmation and regime filters
            # Long: price > R1, above weekly EMA50, volume spike, not choppy
            # Short: price < S1, below weekly EMA50, volume spike, not choppy
            long_entry = (close_val > R1_aligned[i]) and (close_val > ema50_1w_aligned[i]) and volume_spike[i] and not_choppy[i]
            short_entry = (close_val < S1_aligned[i]) and (close_val < ema50_1w_aligned[i]) and volume_spike[i] and not_choppy[i]
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on S1 retracement or if market becomes too choppy
            if close_val < S1_aligned[i] or chop[i] >= 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on R1 retracement or if market becomes too choppy
            if close_val > R1_aligned[i] or chop[i] >= 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_Regime"
timeframe = "1d"
leverage = 1.0
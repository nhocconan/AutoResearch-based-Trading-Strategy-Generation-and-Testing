#!/usr/bin/env python3
"""
12h Donchian20 Breakout + 1d EMA50 Trend + Volume Spike + Chop Filter
Hypothesis: 12h Donchian(20) breakouts capture medium-term momentum. 
Filter: 1d EMA50 trend alignment + volume spike (>2x 20-bar MA) + choppiness regime (CHOP > 61.8 = range, avoid breakouts in chop).
Works in bull/bear via trend filter and discrete sizing (0.25). Targets 50-150 trades over 4 years on 12h.
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
    
    # Load 1d data ONCE before loop for EMA50 trend and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1d Choppiness Index (CHOP) - range: 0-100, >61.8 = range, <38.2 = trending
    # We'll use CHOP > 61.8 to avoid breakouts in choppy markets
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = pd.Series(np.maximum.reduce([
        high_1d[1:] - low_1d[:-1],
        np.abs(high_1d[1:] - close_1d[:-1]),
        np.abs(low_1d[:-1] - close_1d[1:])
    ])).rolling(window=14, min_periods=14).mean().values
    atr_1d = np.concatenate([[np.nan], atr_1d])  # align length
    true_range_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    atr_1d_sum = true_range_sum  # already summed
    max_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = max_high_1d - min_low_1d
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid div by zero
    chop = 100 * np.log10(atr_1d_sum / chop_denom) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Load 12h data ONCE before loop for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h Donchian(20) channels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for 1d EMA50 warmup, Donchian20, and chop
    start_idx = max(60, 21, 34)  # EMA50(50) + Donchian20(20) + Chop14(14) buffers
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1d EMA50
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        # Regime filter: avoid breakouts in choppy markets (CHOP > 61.8 = range)
        not_choppy = chop_aligned[i] <= 61.8
        
        if position == 0:
            # Look for entry signals - require: Donchian breakout + trend + volume + not choppy
            # Long: price breaks above Donchian high AND bullish bias AND volume spike AND not choppy
            long_entry = (curr_high > donchian_high_aligned[i]) and bullish_bias and vol_spike and not_choppy
            # Short: price breaks below Donchian low AND bearish bias AND volume spike AND not choppy
            short_entry = (curr_low < donchian_low_aligned[i]) and bearish_bias and vol_spike and not_choppy
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below Donchian low (invalidates breakout) OR loss of bullish bias
            if (curr_low < donchian_low_aligned[i]) or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Donchian high (invalidates breakout) OR loss of bearish bias
            if (curr_high > donchian_high_aligned[i]) or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike_NoChop"
timeframe = "12h"
leverage = 1.0
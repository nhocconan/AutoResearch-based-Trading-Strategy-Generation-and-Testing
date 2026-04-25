#!/usr/bin/env python3
"""
6h Weekly Pivot + Donchian Breakout + Volume Spike
Hypothesis: Weekly pivot levels (PP, R1, S1) from 1w timeframe identify key support/resistance zones; 
Donchian(20) breakouts in direction of weekly trend (price vs weekly EMA34) with volume confirmation 
capture momentum swings. Designed for 6h timeframe to target 12-37 trades/year (50-150 over 4 years), 
minimizing fee drag. Works in both bull and bear markets by following the weekly trend and avoiding 
counter-trend entries.
"""

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
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # PP = (H + L + C) / 3
    # R1 = (2 * PP) - L
    # S1 = (2 * PP) - H
    # R2 = PP + (H - L)
    # S2 = PP - (H - L)
    # We'll use R1/S1 as primary levels, R2/S2 as stronger breakout
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    pp = typical_price
    r1 = (2 * pp) - df_1w['low']
    s1 = (2 * pp) - df_1w['high']
    r2 = pp + (df_1w['high'] - df_1w['low'])
    s2 = pp - (df_1w['high'] - df_1w['low'])
    
    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp.values)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1.values)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2.values)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2.values)
    
    # Weekly EMA34 for trend filter
    ema_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Donchian(20) channels on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8 * 20-period average (balanced for 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 34)  # Donchian, EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to weekly EMA34
        bullish_bias = curr_close > ema_1w_aligned[i]
        bearish_bias = curr_close < ema_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian high AND above R1/R2 AND bullish bias AND volume spike
            long_entry = (curr_high > donchian_high[i]) and \
                        ((curr_high > r1_aligned[i]) or (curr_high > r2_aligned[i])) and \
                        bullish_bias and vol_spike
            # Short: price breaks below Donchian low AND below S1/S2 AND bearish bias AND volume spike
            short_entry = (curr_low < donchian_low[i]) and \
                         ((curr_low < s1_aligned[i]) or (curr_low < s2_aligned[i])) and \
                         bearish_bias and vol_spike
            
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
            # Exit: price falls below Donchian low (breakdown) OR below S1 (mean reversion to pivot) OR loss of bullish bias
            if (curr_low < donchian_low[i]) or (curr_low < s1_aligned[i]) or (curr_close < ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Donchian high (breakout) OR above R1 (mean reversion to pivot) OR loss of bearish bias
            if (curr_high > donchian_high[i]) or (curr_high > r1_aligned[i]) or (curr_close > ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_DonchianBreakout_VolumeSpike"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and 1w chop regime filter.
- Primary timeframe: 12h for execution, HTF: 1d for Donchian channels and volume MA, 1w for chop regime.
- Chop > 61.8 = ranging (mean revert at Donchian mid), Chop < 38.2 = trending (breakout follow).
- Entry: In trending (Chop < 38.2): Long on breakout above 20-period high, Short on breakdown below 20-period low.
         In ranging (Chop > 61.8): Long at Donchian low with bullish reversal, Short at Donchian high with bearish reversal.
- Volume confirmation: current volume > 1.3 * 20-period volume MA (to filter weak breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 1d
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Get 1w data for chop regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate Chop Index (14-period) on 1w
    # True Range
    tr1 = pd.Series(df_1w['high']).diff().abs()
    tr2 = (pd.Series(df_1w['high']) - pd.Series(df_1w['low'].shift())).abs()
    tr3 = (pd.Series(df_1w['low']) - pd.Series(df_1w['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # High-Low range
    highest_high = pd.Series(df_1w['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1w['low']).rolling(window=14, min_periods=14).min().values
    hl_range = highest_high - lowest_low
    
    # Chop = 100 * log10(atr_sum / hl_range) / log10(14)
    chop = 100 * np.log10(atr_sum / (hl_range + 1e-10)) / np.log10(14)
    
    # Align HTF indicators to 12h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Volume confirmation: current volume > 1.3 * 20-period volume MA (on 12h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Need enough 1w bars for chop and 20 for Donchian/volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        mid = donchian_mid_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if chop_val < 38.2:  # Trending regime: breakout strategy
                    # Bullish breakout: price closes above upper Donchian
                    if curr_close > upper:
                        signals[i] = 0.25
                        position = 1
                    # Bearish breakout: price closes below lower Donchian
                    elif curr_close < lower:
                        signals[i] = -0.25
                        position = -1
                elif chop_val > 61.8:  # Ranging regime: mean reversion at extremes
                    # Long when price touches lower Donchian and shows bullish reversal
                    if curr_low <= lower and curr_close > curr_low:
                        signals[i] = 0.25
                        position = 1
                    # Short when price touches upper Donchian and shows bearish reversal
                    elif curr_high >= upper and curr_close < curr_high:
                        signals[i] = -0.25
                        position = -1
                # In neutral chop (38.2-61.8): no new entries, wait for clearer regime
        elif position == 1:
            # Long exit: price closes below mid OR chop shifts to strong ranging
            if curr_close < mid or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above mid OR chop shifts to strong ranging
            if curr_close > mid or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dVolumeSpike_1wChopRegime_v1"
timeframe = "12h"
leverage = 1.0
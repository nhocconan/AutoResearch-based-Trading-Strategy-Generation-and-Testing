#!/usr/bin/env python3
# 4h_camarilla_pivot_volume_regime_v1
# Hypothesis: 4h strategy using 1d Camarilla pivot levels (L3, L4, H3, H4) for institutional support/resistance. Enters long when price touches L3/L4 with volume confirmation (>1.5x 20-bar avg volume) in trending regime (CHOP < 38.2); short when price touches H3/H4 with same filters. Exits on opposite pivot touch. Uses 1d HTF for pivot calculation. Target: 19-50 trades/year via tight pivot-level entries + volume + regime filters. Works in bull/bear by fading extreme moves at proven institutional levels with volume confirmation, avoiding whipsaws via chop regime filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period = 20 * 4h bars)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Multi-timeframe: 1d Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla formulas
    range_ = prev_high - prev_low
    h3 = prev_close + range_ * 1.1 / 4
    l3 = prev_close - range_ * 1.1 / 4
    h4 = prev_close + range_ * 1.1 / 2
    l4 = prev_close - range_ * 1.1 / 2
    
    # Align HTF levels to LTF (each 1d bar = 6 * 4h bars)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3.values)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3.values)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4.values)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4.values)
    
    # Choppiness Index (14-period) for regime filter
    def calculate_chop(high, low, close, window=14):
        tr = np.zeros(len(close))
        tr[0] = high[0] - low[0]
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        sum_atr = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        range_max_min = highest_high - lowest_low
        chop = np.full_like(sum_atr, np.nan, dtype=float)
        valid = (range_max_min != 0) & (~np.isnan(sum_atr)) & (~np.isnan(range_max_min))
        chop[valid] = 100 * np.log10(sum_atr[valid] / np.log10(window) / range_max_min[valid])
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Regime filter: trending only (CHOP < 38.2)
        trending_regime = chop[i] < 38.2
        
        if position == 1:  # Long position
            # Exit: price touches or breaks H3/H4 level (opposite side)
            if close[i] >= h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches or breaks L3/L4 level (opposite side)
            if close[i] <= l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter only in trending regime with volume confirmation
            if trending_regime and volume_confirmed:
                # Long: price touches L3 or L4 support (bullish bounce)
                if close[i] <= l4_aligned[i] and close[i] >= l3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price touches H3 or H4 resistance (bearish rejection)
                elif close[i] >= h3_aligned[i] and close[i] <= h4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
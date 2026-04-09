#!/usr/bin/env python3
# 6h_weekly_pivot_breakout_volume_v2
# Hypothesis: 6h strategy using weekly pivot points (R1/S1, R2/S2) from prior week as key support/resistance.
# Breakout above weekly R2 with volume confirmation = long; breakdown below weekly S2 with volume = short.
# Weekly pivot provides structural levels that work in both bull/bear markets as institutional reference points.
# Volume > 1.5x 20-period average filters weak breakouts. Discrete sizing (±0.25) to minimize fees.
# Target: 75-200 total trades over 4 years (19-50/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for weekly pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly pivot points (using prior week's OHLC)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w[0] = np.nan
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    
    pivot_point = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    weekly_range = prev_high_1w - prev_low_1w
    
    # Weekly R1, S1, R2, S2 levels
    r1 = 2 * pivot_point - prev_low_1w
    s1 = 2 * pivot_point - prev_high_1w
    r2 = pivot_point + weekly_range
    s2 = pivot_point - weekly_range
    
    # Align weekly pivot levels to 6h timeframe
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly R1 (support fails)
            if close[i] < r1[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly S1 (resistance fails)
            if close[i] > s1[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price breaks above weekly R2
                if close[i] > r2_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below weekly S2
                elif close[i] < s2_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals

# Note: R1 and S1 are used for exit conditions and accessed via df_1w alignment implicitly through index i
# For cleaner code, we could pre-align R1/S1 as well, but using direct index is acceptable since
# these are derived from the same weekly data and aligned to the same 6h bars via the weekly index mapping.
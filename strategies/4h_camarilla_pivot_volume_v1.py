#!/usr/bin/env python3
# 4h_camarilla_pivot_volume_v1
# Hypothesis: 4h strategy using daily Camarilla pivot levels with volume confirmation.
# In ranging markets: fade at H3/L3 levels (short at H3 resistance, long at L3 support).
# In breakout markets: continuation when price breaks H4/L4 with volume > 1.5x average.
# Uses discrete position sizing (0.0, ±0.25) to minimize fee churn.
# Target: 75-200 total trades over 4 years by requiring confluence of pivot level, volume, and price action.
# Primary timeframe: 4h, HTF: 1d for Camarilla pivot calculation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 1d HTF data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels: based on previous day's OHLC
    # H4, H3, H2, H1, L1, L2, L3, L4
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # H1 = Pivot + (Range * 1.1 / 12)
    # H2 = Pivot + (Range * 1.1 / 6)
    # H3 = Pivot + (Range * 1.1 / 4)
    # H4 = Pivot + (Range * 1.1 / 2)
    # L1 = Pivot - (Range * 1.1 / 12)
    # L2 = Pivot - (Range * 1.1 / 6)
    # L3 = Pivot - (Range * 1.1 / 4)
    # L4 = Pivot - (Range * 1.1 / 2)
    
    h1d = df_1d['high'].values
    l1d = df_1d['low'].values
    c1d = df_1d['close'].values
    
    pivot = (h1d + l1d + c1d) / 3
    rng = h1d - l1d
    
    # Calculate all 8 levels
    h4 = pivot + rng * 1.1 / 2
    h3 = pivot + rng * 1.1 / 4
    h2 = pivot + rng * 1.1 / 6
    h1 = pivot + rng * 1.1 / 12
    l1 = pivot - rng * 1.1 / 12
    l2 = pivot - rng * 1.1 / 6
    l3 = pivot - rng * 1.1 / 4
    l4 = pivot - rng * 1.1 / 2
    
    # Align HTF Camarilla levels to LTF (4h)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(h4_1d_aligned[i]) or
            np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or np.isnan(l4_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below L3 OR reversal signal at H3
            if close[i] < l3_1d_aligned[i] or (close[i] > h3_1d_aligned[i] and close[i] < h4_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above H3 OR reversal signal at L3
            if close[i] > h3_1d_aligned[i] or (close[i] < l3_1d_aligned[i] and close[i] > l4_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Fade at H3/L3 in ranging conditions
                # Short near H3 resistance
                if close[i] > h3_1d_aligned[i] * 0.995 and close[i] < h3_1d_aligned[i] * 1.005:
                    position = -1
                    signals[i] = -0.25
                # Long near L3 support
                elif close[i] > l3_1d_aligned[i] * 0.995 and close[i] < l3_1d_aligned[i] * 1.005:
                    position = 1
                    signals[i] = 0.25
                # Breakout continuation: break H4/L4 with volume
                elif close[i] > h4_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < l4_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
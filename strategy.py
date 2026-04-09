#!/usr/bin/env python3
# 4h_12h_camarilla_volume_reversion_v1
# Hypothesis: Camarilla pivot levels on 12h provide strong support/resistance levels. Price reversals from these levels with volume confirmation work in both bull and bear markets. Uses 12h Camarilla levels (H4, L4) for entry, volume > 1.5x 20-period average for confirmation, and exits on opposite level touch or volume fade. Position size 0.25. Target: 20-30 trades/year (80-120 over 4 years) to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_volume_reversion_v1"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for 12h: H4, L4 (key reversal levels)
    # Camarilla: H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2
    hl_range = df_12h['high'] - df_12h['low']
    camarilla_h4 = df_12h['close'] + 1.1 * hl_range / 2
    camarilla_l4 = df_12h['close'] - 1.1 * hl_range / 2
    
    # Align to 4h timeframe (wait for completed 12h bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4.values)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4.values)
    
    # Volume confirmation: 20-period average on 4h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches L4 (support) or volume drops below average
            if low[i] <= camarilla_l4_aligned[i] or volume[i] < vol_ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches H4 (resistance) or volume drops below average
            if high[i] >= camarilla_h4_aligned[i] or volume[i] < vol_ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price touches L4 with volume confirmation (bounce from support)
            if low[i] <= camarilla_l4_aligned[i] and volume[i] > vol_threshold[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price touches H4 with volume confirmation (rejection at resistance)
            elif high[i] >= camarilla_h4_aligned[i] and volume[i] > vol_threshold[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
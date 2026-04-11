#!/usr/bin/env python3
# 4h_1d_camarilla_volume_breakout_v1
# Strategy: 4h breakout at Camarilla pivot levels (H3/L3) from 1d timeframe with volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (H3/L3) act as strong support/resistance. Breakouts above H3 or below L3
# with volume > 1.5x 20-period average capture institutional flow. Works in both bull/bear markets
# as breakouts capture momentum in trending regimes and reversals in ranging markets.
# Designed for low trade frequency (<30/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_volume_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar
    # Formula: 
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.0 * (high - low)
    # L3 = close - 1.0 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # We use H3 and L3 as key levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h3 = close_1d + 1.0 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.0 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: 1d volume > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or \
           np.isnan(vol_avg_20_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_confirm = volume_1d[i] > 1.5 * vol_avg_20_1d[i]  # Use raw 1d volume for current bar
        
        # Price levels
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        price = close[i]
        
        # Entry conditions
        # Long: Price breaks above H3 with volume confirmation
        if price > h3 and vol_confirm and position != 1:
            # Additional check: ensure we didn't just break above H3 in previous bar
            if i == 50 or close[i-1] <= camarilla_h3_aligned[i-1]:
                position = 1
                signals[i] = 0.25
        # Short: Price breaks below L3 with volume confirmation
        elif price < l3 and vol_confirm and position != -1:
            # Additional check: ensure we didn't just break below L3 in previous bar
            if i == 50 or close[i-1] >= camarilla_l3_aligned[i-1]:
                position = -1
                signals[i] = -0.25
        # Exit: Price returns to the opposite level (mean reversion)
        elif position == 1 and price < camarilla_l3_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and price > camarilla_h3_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
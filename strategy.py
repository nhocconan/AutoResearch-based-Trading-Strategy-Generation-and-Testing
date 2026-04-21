#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 12h Camarilla pivot levels with volume confirmation.
In both bull and bear markets, price reacts strongly at Camarilla levels (R3/S3 for reversal, 
R4/S4 for breakout). Uses 12h Camarilla for structural levels and 6h volume spike for confirmation.
Target: 15-35 trades/year to minimize fee drag on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (using previous day's high-low-close)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla calculation: based on previous period's range
    # R4 = Close + (High - Low) * 1.1/2
    # R3 = Close + (High - Low) * 1.1/4
    # S3 = Close - (High - Low) * 1.1/4
    # S4 = Close - (High - Low) * 1.1/2
    rng = high_12h - low_12h
    r4 = close_12h + rng * 1.1 / 2
    r3 = close_12h + rng * 1.1 / 4
    s3 = close_12h - rng * 1.1 / 4
    s4 = close_12h - rng * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # Volume confirmation: 6h volume / 20-period average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        vol_threshold = 1.5  # Volume spike filter
        
        if position == 0:
            # Enter long: price at S3/S4 with volume spike
            # S3: potential reversal zone (go long if price holds above S3)
            # S4: breakout level (go long if price breaks above S4 with volume)
            if ((price_close > s3_aligned[i] * 1.001 and price_close < s3_aligned[i] * 1.02) or
                (price_close > s4_aligned[i] and vol_ratio_val > vol_threshold)):
                signals[i] = 0.25
                position = 1
            # Enter short: price at R3/R4 with volume spike
            # R3: potential reversal zone (go short if price holds below R3)
            # R4: breakdown level (go short if price breaks below R4 with volume)
            elif ((price_close < r3_aligned[i] * 0.999 and price_close > r3_aligned[i] * 0.98) or
                  (price_close < r4_aligned[i] and vol_ratio_val > vol_threshold)):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price reaches opposite level or volume drops
            if position == 1:
                # Exit long: price reaches R3 (resistance) or volume drops significantly
                if price_close >= r3_aligned[i] * 0.995 or vol_ratio_val < 0.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price reaches S3 (support) or volume drops significantly
                if price_close <= s3_aligned[i] * 1.005 or vol_ratio_val < 0.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_Pivot_Volume_Spike"
timeframe = "6h"
leverage = 1.0
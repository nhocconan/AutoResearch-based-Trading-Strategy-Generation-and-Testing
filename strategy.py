#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout with volume confirmation and weekly ATR regime filter.
Long when price breaks above Camarilla R1 AND 12h volume > 2.0x 20-bar avg AND weekly ATR(14) > 20-bar ATR(14) MA (expanding volatility).
Short when price breaks below Camarilla S1 AND 12h volume > 2.0x 20-bar avg AND weekly ATR(14) > 20-bar ATR(14) MA.
Exit when price touches Camarilla H5/L5 or opposite breakout level.
Uses 1w for ATR-based volatility regime and 12h for execution, volume, and Camarilla levels.
Designed to capture breakouts during expanding volatility regimes across bull and bear markets.
Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for ATR regime filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ATR (14-period)
    tr1 = np.maximum(high_1w - low_1w, 
                     np.absolute(high_1w - np.roll(close_1w, 1)),
                     np.absolute(low_1w - np.roll(close_1w, 1)))
    tr1[0] = high_1w[0] - low_1w[0]
    atr_1w = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    atr_ma_20 = pd.Series(atr_1w).rolling(window=20, min_periods=20).mean().values
    atr_expanding = atr_1w > atr_ma_20  # volatility expanding
    
    # Get 12h data for Camarilla levels and volume
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h Camarilla levels (based on previous bar)
    # Camarilla: H5 = C + 1.1*(H-L)*1.1/2, L5 = C - 1.1*(H-L)*1.1/2
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # R2 = C + (H-L)*1.1/6, S2 = C - (H-L)*1.1/6
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    # H5/L5 = extreme levels
    range_12h = high_12h - low_12h
    camarilla_r1 = close_12h + range_12h * 1.1 / 12
    camarilla_s1 = close_12h - range_12h * 1.1 / 12
    camarilla_h5 = close_12h + range_12h * 1.1 * 1.1 / 2
    camarilla_l5 = close_12h - range_12h * 1.1 * 1.1 / 2
    
    # Calculate 12h volume MA (20-period)
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe (12h)
    atr_expanding_aligned = align_htf_to_ltf(prices, df_1w, atr_expanding)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h5)
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l5)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_expanding_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_h5_aligned[i]) or
            np.isnan(camarilla_l5_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 2.0x 20-bar average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20_aligned[i]
        
        # Regime filter: ATR expanding indicates increasing volatility
        volatility_expanding = atr_expanding_aligned[i]
        
        # Breakout conditions
        breakout_r1 = close[i] > camarilla_r1_aligned[i]
        breakout_s1 = close[i] < camarilla_s1_aligned[i]
        
        # Exit conditions: touch H5/L5 or opposite breakout level
        touch_h5 = close[i] >= camarilla_h5_aligned[i]
        touch_l5 = close[i] <= camarilla_l5_aligned[i]
        touch_opposite = (position == 1 and close[i] < camarilla_s1_aligned[i]) or \
                         (position == -1 and close[i] > camarilla_r1_aligned[i])
        
        if position == 0:
            # Long: break above R1 with volume confirmation and expanding volatility
            if (breakout_r1 and volume_confirmed and volatility_expanding):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume confirmation and expanding volatility
            elif (breakout_s1 and volume_confirmed and volatility_expanding):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch H5 or break below S1
            if (touch_h5 or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch L5 or break above R1
            if (touch_l5 or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Volume_ATR_Expanding"
timeframe = "12h"
leverage = 1.0
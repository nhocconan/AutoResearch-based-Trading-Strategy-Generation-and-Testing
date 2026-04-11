#!/usr/bin/env python3
"""
4h_1d_camarilla_breakout_volume_v2
Strategy: 4h Camarilla pivot breakout with volume confirmation
Timeframe: 4h
Leverage: 1.0
Hypothesis: Uses daily Camarilla pivot levels (H3/L3) for breakout entries on the 4h chart with volume confirmation (>1.3x average volume). Designed to capture institutional breakouts in trending markets while avoiding false breakouts. Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_v2"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.3 * vol_avg)
    
    # Calculate Camarilla levels from daily data
    # Camarilla levels: H3 = Close + 1.1*(High-Low)/2, L3 = Close - 1.1*(High-Low)/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_H3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_L3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(vol_avg[i]) or 
            np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Breakout conditions using Camarilla H3/L3 levels
        breakout_up = price_close > camarilla_H3_aligned[i]
        breakout_down = price_close < camarilla_L3_aligned[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Trading logic
        if breakout_up and vol_confirmed and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_down and vol_confirmed and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and not vol_confirmed:
            # Exit when volume drops (momentum fading)
            position = 0
            signals[i] = 0.0
        elif position == -1 and not vol_confirmed:
            # Exit when volume drops (momentum fading)
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
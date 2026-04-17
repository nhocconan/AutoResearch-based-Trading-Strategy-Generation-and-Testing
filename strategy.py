#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1w EMA50 trend filter and volume confirmation.
Long when Bull Power > 0 and Bear Power < 0 (bullish momentum) and price > 1w EMA50.
Short when Bear Power < 0 and Bull Power > 0 (bearish momentum) and price < 1w EMA50.
Elder Ray measures bull/bear strength relative to EMA13, works in both bull and bear markets by measuring power dynamics.
1w EMA50 provides major trend filter to avoid counter-trend trades.
Volume confirmation ensures institutional participation.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 13-period EMA for Elder Ray (using 6h data)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_ok = volume[i] > vol_ma * 0.8  # Allow 20% below average
        else:
            volume_ok = True
        
        if position == 0:
            # Long: Bull Power > 0 (strong buying), Bear Power < 0 (weak selling), price above 1w EMA50, volume confirmation
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                close[i] > ema50_1w_aligned[i] and 
                volume_ok):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (strong selling), Bull Power > 0 (weak buying), price below 1w EMA50, volume confirmation
            elif (bear_power[i] < 0 and 
                  bull_power[i] > 0 and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power turns negative OR Bear Power turns positive OR price crosses below 1w EMA50
            if (bull_power[i] <= 0 or 
                bear_power[i] >= 0 or 
                close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power turns positive OR Bull Power turns negative OR price crosses above 1w EMA50
            if (bear_power[i] >= 0 or 
                bull_power[i] <= 0 or 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1wElderRay_EMA50_Volume"
timeframe = "6h"
leverage = 1.0
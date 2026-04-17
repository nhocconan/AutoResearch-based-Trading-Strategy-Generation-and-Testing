#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 12h EMA regime filter.
Long when Bull Power > 0 AND Bear Power < 0 AND price > 12h EMA50 (bullish regime).
Short when Bear Power < 0 AND Bull Power < 0 AND price < 12h EMA50 (bearish regime).
Exit when power signs diverge or price crosses 12h EMA50.
Uses 6h for Elder Ray calculation and 12h for EMA regime to reduce whipsaw.
Target: 50-150 total trades over 4 years (12-37/year). Elder Ray measures bull/bear strength,
EMA filter ensures trading with higher timeframe trend, reducing counter-trend losses.
Works in bull markets (captures uptrends via Bull Power) and bear markets (captures downtrends via Bear Power).
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
    
    # Get 6h data for Elder Ray calculation
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Elder Ray on 6h timeframe (13-period EMA)
    close_6h_series = pd.Series(close_6h)
    ema13 = close_6h_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_6h - ema13  # Bull Power = High - EMA13
    bear_power = low_6h - ema13   # Bear Power = Low - EMA13
    
    # Get 12h data for EMA50 regime filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 6h Elder Ray to 6h timeframe (no alignment needed)
    bull_power_aligned = bull_power
    bear_power_aligned = bear_power
    
    # Align 12h EMA50 to 6h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        bp = bull_power_aligned[i]
        br = bear_power_aligned[i]
        ema50 = ema50_12h_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND price > 12h EMA50
            if bp > 0 and br < 0 and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power < 0 AND price < 12h EMA50
            elif br < 0 and bp < 0 and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power <= 0 OR Bear Power >= 0 OR price < 12h EMA50
            if bp <= 0 or br >= 0 or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power >= 0 OR Bull Power >= 0 OR price > 12h EMA50
            if br >= 0 or bp >= 0 or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_12hEMA50_Regime"
timeframe = "6h"
leverage = 1.0
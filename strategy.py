#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) + 12h EMA50 Regime Filter.
Long when Bull Power > 0 AND price > 12h EMA50 (bullish regime).
Short when Bear Power < 0 AND price < 12h EMA50 (bearish regime).
Exit when opposing Elder Ray power becomes positive (Bull Power < 0 for longs, Bear Power > 0 for shorts) or regime flip.
Uses 6h for Elder Ray calculation, 12h for EMA50 trend filter.
Target: 50-150 total trades over 4 years (12-37/year). Elder Ray measures bull/bear strength via EMA13, 
12h EMA50 filters for higher-timeframe trend alignment to reduce false signals in chop and align with major trends.
Works in bull markets (captures strength) and bear markets (avoids longs in downtrend, takes shorts in downtrend strength).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 6h data for Elder Ray calculation
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Elder Ray on 6h timeframe
    # Bull Power = High - EMA13(Close)
    # Bear Power = Low - EMA13(Close)
    close_6h_series = pd.Series(close_6h)
    ema13_6h = close_6h_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_6h - ema13_6h
    bear_power = low_6h - ema13_6h
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 6h Elder Ray to 6h timeframe (no alignment needed as we're already on 6h)
    bull_power_aligned = bull_power  # Already on 6h timeframe
    bear_power_aligned = bear_power  # Already on 6h timeframe
    
    # Align 12h EMA50 to 6h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(ema50_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        bp = bull_power_aligned[i]
        br = bear_power_aligned[i]
        price = close[i]
        ema50 = ema50_12h_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 (bullish strength) AND price > 12h EMA50 (bullish regime)
            if bp > 0 and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bearish strength) AND price < 12h EMA50 (bearish regime)
            elif br < 0 and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power < 0 (lost bullish strength) OR price < 12h EMA50 (regime flip to bearish)
            if bp < 0 or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power > 0 (lost bearish strength) OR price > 12h EMA50 (regime flip to bullish)
            if br > 0 or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_12hEMA50_Regime"
timeframe = "6h"
leverage = 1.0
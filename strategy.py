#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 filter
# Long when Bull Power > 0, Bear Power < 0, and price > 1d EMA50 (bullish bias)
# Short when Bear Power < 0, Bull Power < 0, and price < 1d EMA50 (bearish bias)
# Exit when Elder Power signals reverse or price crosses 1d EMA50
# Elder Ray measures bull/bear power relative to EMA; 1d EMA50 provides trend filter
# Designed to work in both bull and bear markets by following the 1d trend
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA13 for Elder Ray (13-period)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean()
    
    # Calculate Bull Power and Bear Power
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate EMA50 on 1d for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean()
    ema50_1d_values = ema50_1d.values
    
    # Align 1d EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d_values)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema13_val = ema13[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        ema50_val = ema50_1d_aligned[i]
        close_val = close[i]
        
        if position == 0:
            # Long setup: Bull Power > 0, Bear Power < 0, price > 1d EMA50
            if (bull_val > 0 and bear_val < 0 and close_val > ema50_val):
                position = 1
                signals[i] = position_size
            # Short setup: Bear Power < 0, Bull Power < 0, price < 1d EMA50
            elif (bear_val < 0 and bull_val < 0 and close_val < ema50_val):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bear Power > 0 OR price < 1d EMA50
            if (bear_val > 0 or close_val < ema50_val):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bull Power > 0 OR price > 1d EMA50
            if (bull_val > 0 or close_val > ema50_val):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_ElderRay_1dEMA50"
timeframe = "6h"
leverage = 1.0
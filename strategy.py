#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with 1d regime filter
# Elder Ray (Bull/Bear Power) measures bull/bear strength relative to EMA.
# Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# In bull regime (price > weekly EMA50), go long when Bull Power crosses above zero.
# In bear regime (price < weekly EMA50), go short when Bear Power crosses below zero.
# Uses 1d EMA13 for Bull/Bear Power and weekly EMA50 for regime filter.
# Target: 15-30 trades/year per symbol.
name = "6h_ElderRay_1dEMA13_WeeklyEMA50_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for calculations (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    # Get weekly data for regime filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA13 on daily closes for Elder Ray
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Align EMA13 to 6h timeframe
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Calculate Bull Power and Bear Power
    bull_power = high - ema13_1d_aligned
    bear_power = low - ema13_1d_aligned
    
    # Calculate weekly EMA50 for regime filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align weekly EMA50 to 6h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema13_1d_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: bull if price > weekly EMA50, bear if price < weekly EMA50
        is_bull_regime = close[i] > ema50_1w_aligned[i]
        is_bear_regime = close[i] < ema50_1w_aligned[i]
        
        if position == 0:
            # Long: bull regime AND bull power crosses above zero
            if is_bull_regime and bull_power[i] > 0 and bull_power[i-1] <= 0:
                signals[i] = 0.25
                position = 1
            # Short: bear regime AND bear power crosses below zero
            elif is_bear_regime and bear_power[i] < 0 and bear_power[i-1] >= 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bear power crosses below zero (momentum shift)
            if bear_power[i] < 0 and bear_power[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bull power crosses above zero (momentum shift)
            if bull_power[i] > 0 and bull_power[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
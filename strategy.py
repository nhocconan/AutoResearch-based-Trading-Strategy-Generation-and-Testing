#!/usr/bin/env python3
"""
12H_Camarilla_Pivot_Squeeze: Trade reversals at Camarilla H3/L3 levels during low volatility.
Long when price touches L3 during Bollinger squeeze (low volatility) and closes above L3.
Short when price touches H3 during Bollinger squeeze and closes below H3.
Exit when price reaches the opposite H3/L3 level or volatility expands.
Uses 1-day Camarilla levels for structure and Bollinger Bands for volatility regime.
Designed for low trade frequency by requiring volatility contraction + extreme level touch.
Works in ranging markets (2025+) by capturing mean reversion at statistical extremes.
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
    volume = prices['volume'].values
    
    # Load 1-day data for Camarilla pivot levels - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels using previous day's data
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    # H3/L3 are the strongest reversal levels
    h3_1d = close_1d + (high_1d - low_1d) * 1.1 / 4.0
    l3_1d = close_1d - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align to 12h timeframe
    pp_12h = align_htf_to_ltf(prices, df_1d, pp_1d)
    h3_12h = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_12h = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # Bollinger Bands (20, 2) for volatility regime - calculated on 12h close
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean()
    bb_std = close_series.rolling(window=20, min_periods=20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # Bollinger squeeze: width below 20-period average indicates low volatility
    bb_width_ma = bb_width.rolling(window=20, min_periods=20).mean()
    squeeze = bb_width < bb_width_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after enough data for BB
        # Skip if data not ready
        if (np.isnan(pp_12h[i]) or np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or
            np.isnan(bb_width[i]) or np.isnan(bb_width_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price touches L3 during squeeze and closes above it
            if (low[i] <= l3_12h[i] and close[i] > l3_12h[i] and squeeze[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price touches H3 during squeeze and closes below it
            elif (high[i] >= h3_12h[i] and close[i] < h3_12h[i] and squeeze[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price reaches H3 or volatility expands (squeeze ends)
                if high[i] >= h3_12h[i] or not squeeze[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price reaches L3 or volatility expands
                if low[i] <= l3_12h[i] or not squeeze[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_Pivot_Squeeze"
timeframe = "12h"
leverage = 1.0
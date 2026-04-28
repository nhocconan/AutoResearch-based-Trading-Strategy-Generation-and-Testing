#!/usr/bin/env python3
"""
6h_Aroon_Trend_WeeklyPivot_Filter
Hypothesis: Use Aroon oscillator (25-period) to detect strong trends on 6h, filtered by weekly pivot direction (bullish/bearish bias from weekly high/low). 
Aroon > 50 indicates bullish momentum, Aroon < -50 bearish. Weekly pivot adds higher timeframe bias to avoid counter-trend trades.
Works in bull markets by catching strong uptrends with bullish weekly bias, and in bear markets by catching strong downtrends with bearish weekly bias.
Targets 15-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate Aroon oscillator (25-period) on 6h data
    period = 25
    high_period = pd.Series(high).rolling(window=period, min_periods=period).apply(lambda x: x.argmax(), raw=True)
    low_period = pd.Series(low).rolling(window=period, min_periods=period).apply(lambda x: x.argmin(), raw=True)
    aroon_up = ((period - high_period) / period) * 100
    aroon_down = ((period - low_period) / period) * 100
    aroon_osc = aroon_up - aroon_down  # ranges from -100 to 100
    
    # Get weekly high/low for pivot bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Weekly bias: 1 if close > weekly midpoint (bullish), -1 if close < weekly midpoint (bearish)
    weekly_mid = (weekly_high + weekly_low) / 2.0
    weekly_bias = np.where(close[-1] > weekly_mid[-1], 1, -1)  # using latest weekly close for bias
    # For historical alignment, we need to compute bias per week
    # Simplify: use weekly close > weekly midpoint as bullish bias for that week
    weekly_close = df_1w['close'].values
    weekly_bias_raw = np.where(weekly_close > weekly_mid, 1, -1)
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias_raw)
    
    # Aroon oscillator aligned to 6h (already on 6h, no alignment needed)
    aroon_osc_aligned = aroon_osc  # same index as prices
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(aroon_osc_aligned[i]) or np.isnan(weekly_bias_aligned[i]):
            signals[i] = 0.0
            continue
        
        aroon = aroon_osc_aligned[i]
        bias = weekly_bias_aligned[i]
        
        # Entry conditions
        # Long: strong bullish momentum (Aroon > 50) + bullish weekly bias
        long_entry = aroon > 50 and bias == 1
        # Short: strong bearish momentum (Aroon < -50) + bearish weekly bias
        short_entry = aroon < -50 and bias == -1
        
        # Exit conditions: momentum weakening or bias flip
        long_exit = aroon < 0 or bias == -1
        short_exit = aroon > 0 or bias == 1
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Aroon_Trend_WeeklyPivot_Filter"
timeframe = "6h"
leverage = 1.0
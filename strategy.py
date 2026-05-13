#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Strategy
Hypothesis: Camarilla pivot levels from daily timeframe act as strong support/resistance.
Price rejection at these levels with volume confirmation and trend alignment provides
high-probability mean-reversion entries. Designed for low trade frequency (15-25/year)
to minimize fee drag while capturing reversals in both bull and bear markets.
"""

name = "12h_Camarilla_Pivot_Strategy"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's range
    # Camarilla: H4 = Close + 1.5*(High-Low), L4 = Close - 1.5*(High-Low)
    # Using previous day's data to avoid look-ahead
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Calculate resistance and support levels
    R4 = prev_close + 1.5 * prev_range  # Strong resistance
    R3 = prev_close + 1.25 * prev_range  # Resistance
    S3 = prev_close - 1.25 * prev_range  # Support
    S4 = prev_close - 1.5 * prev_range   # Strong support
    
    # Align levels to 12h timeframe (wait for daily close)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Trend filter: 50-period EMA on 1-day timeframe
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Price touches/slightly breaks S3/S4 with rejection and volume
            # In downtrend (price < EMA50), look for longs at support
            if close[i] < ema_50_1d_aligned[i]:  # Downtrend filter
                if low[i] <= S3_aligned[i] and close[i] > S3_aligned[i] * 0.999:  # Touch S3 with close recovery
                    if volume_confirm[i]:
                        signals[i] = 0.25
                        position = 1
                elif low[i] <= S4_aligned[i] and close[i] > S4_aligned[i] * 0.999:  # Touch S4 with close recovery
                    if volume_confirm[i]:
                        signals[i] = 0.25
                        position = 1
            # SHORT: Price touches/slightly breaks R3/R4 with rejection and volume
            # In uptrend (price > EMA50), look for shorts at resistance
            elif close[i] > ema_50_1d_aligned[i]:  # Uptrend filter
                if high[i] >= R3_aligned[i] and close[i] < R3_aligned[i] * 1.001:  # Touch R3 with close rejection
                    if volume_confirm[i]:
                        signals[i] = -0.25
                        position = -1
                elif high[i] >= R4_aligned[i] and close[i] < R4_aligned[i] * 1.001:  # Touch R4 with close rejection
                    if volume_confirm[i]:
                        signals[i] = -0.25
                        position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches midpoint or shows weakness
            midpoint = (S3_aligned[i] + R3_aligned[i]) / 2
            if close[i] >= midpoint or close[i] < S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches midpoint or shows strength
            midpoint = (S3_aligned[i] + R3_aligned[i]) / 2
            if close[i] <= midpoint or close[i] > R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
1d_1w_Camarilla_Pivot_Breakout_Volume
Hypothesis: Weekly Camarilla pivot levels provide strong support/resistance on the daily chart.
Breakouts above R4 or below S4 with volume expansion and trend alignment (via 200-day EMA)
capture institutional participation. Works in both bull and bear markets by trading with the
dominant trend. Target: 15-25 trades/year per symbol.
"""

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
    
    # Get weekly data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each weekly bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla formulas
    close_prev = np.roll(close_1w, 1)
    close_prev[0] = close_1w[0]  # first bar uses its own close
    
    range_1w = high_1w - low_1w
    
    # Resistance levels
    R1 = close_prev + (range_1w * 1.0833 / 12)
    R2 = close_prev + (range_1w * 1.1666 / 6)
    R3 = close_prev + (range_1w * 1.2500 / 4)
    R4 = close_prev + (range_1w * 1.5000 / 2)
    
    # Support levels
    S1 = close_prev - (range_1w * 1.0833 / 12)
    S2 = close_prev - (range_1w * 1.1666 / 6)
    S3 = close_prev - (range_1w * 1.2500 / 4)
    S4 = close_prev - (range_1w * 1.5000 / 2)
    
    # Align levels to daily timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1w, R2)
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    R4_aligned = align_htf_to_ltf(prices, df_1w, R4)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1w, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1w, S4)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.8)
    
    # Trend filter: 200-day EMA
    close_series = pd.Series(close)
    ema_200 = close_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(200, n):
        # Skip if any required data is not ready
        if (np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(volume_expansion[i]) or np.isnan(ema_200[i])):
            signals[i] = 0.0
            continue
        
        # Long breakout: price breaks above R4 with volume expansion and above EMA200
        long_breakout = close[i] > R4_aligned[i] and volume_expansion[i] and close[i] > ema_200[i]
        
        # Short breakdown: price breaks below S4 with volume expansion and below EMA200
        short_breakout = close[i] < S4_aligned[i] and volume_expansion[i] and close[i] < ema_200[i]
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1d_1w_Camarilla_Pivot_Breakout_Volume"
timeframe = "1d"
leverage = 1.0
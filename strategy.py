#!/usr/bin/env python3
"""
6h_MultiTF_Ichimoku_Cloud_Breakout
Hypothesis: Combines 6h Ichimoku Tenkan/Kijun cross with 1d cloud color as trend filter and volume confirmation.
In bull markets, price above cloud with bullish TK cross signals continuation; in bear markets, price below cloud with bearish TK cross signals continuation.
The 1d cloud provides higher timeframe trend context, reducing false signals during range-bound periods.
Volume surge confirms breakout strength. Targets 15-30 trades/year with discrete sizing (0.25) to minimize fee churn.
Works in both bull and bear by following the higher timeframe trend direction.
"""

name = "6h_MultiTF_Ichimoku_Cloud_Breakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for Ichimoku cloud (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)

    # Calculate Ichimoku components on 1d data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Volume confirmation: 24-period average on 6h (4 trading days)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # Start from 52 to have enough data for all indicators
        # Get aligned values for current 6h bar
        tenkan = tenkan_aligned[i]
        kijun = kijun_aligned[i]
        senkou_a = senkou_a_aligned[i]
        senkou_b = senkou_b_aligned[i]
        vol_avg = vol_avg_24[i]
        
        # Skip if any required data is NaN
        if (np.isnan(tenkan) or np.isnan(kijun) or np.isnan(senkou_a) or 
            np.isnan(senkou_b) or np.isnan(vol_avg)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Determine cloud color and price position relative to cloud
        # Green cloud (bullish): Senkou A > Senkou B
        # Red cloud (bearish): Senkou A < Senkou B
        green_cloud = senkou_a > senkou_b
        red_cloud = senkou_a < senkou_b
        above_cloud = close[i] > max(senkou_a, senkou_b)
        below_cloud = close[i] < min(senkou_a, senkou_b)
        
        # TK cross signals
        tk_bullish_cross = tenkan > kijun and tenkan <= kijun  # Current bullish, previous was bearish or equal
        tk_bearish_cross = kijun > tenkan and kijun <= tenkan  # Current bearish, previous was bullish or equal
        # Fix crossover detection
        if i > 52:
            tk_bullish_cross = tenkan > kijun and tenkan_aligned[i-1] <= kijun_aligned[i-1]
            tk_bearish_cross = kijun > tenkan and kijun_aligned[i-1] <= tenkan_aligned[i-1]
        else:
            tk_bullish_cross = False
            tk_bearish_cross = False

        if position == 0:
            # LONG: Price above green cloud + bullish TK cross + volume surge
            if (above_cloud and green_cloud and tk_bullish_cross and 
                volume[i] > vol_avg * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below red cloud + bearish TK cross + volume surge
            elif (below_cloud and red_cloud and tk_bearish_cross and 
                  volume[i] > vol_avg * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below cloud or bearish TK cross
            if (close[i] < senkou_a or close[i] < senkou_b or tk_bearish_cross):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above cloud or bullish TK cross
            if (close[i] > senkou_a or close[i] > senkou_b or tk_bullish_cross):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
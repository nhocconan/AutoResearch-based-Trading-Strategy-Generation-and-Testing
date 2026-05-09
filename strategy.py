#!/usr/bin/env python3
# Hypothesis: 12h timeframe with 1-day Williams %R for overbought/oversold conditions and 1-week EMA trend filter.
# Enters long when Williams %R < -80 (oversold) and price > 1-week EMA, short when Williams %R > -20 (overbought) and price < 1-week EMA.
# Exits when Williams %R returns to neutral range (-80 to -20) or price crosses EMA in opposite direction.
# Williams %R identifies exhaustion points in both bull and bear markets, while EMA filter ensures trades align with higher-timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "12h_WilliamsR_EMA_Trend"
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
    
    # Calculate 1-day Williams %R(14)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Williams %R conditions: oversold < -80, overbought > -20
    williams_oversold = williams_r < -80
    williams_overbought = williams_r > -20
    
    # Calculate 1-week EMA(20) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    ema_20 = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align indicators to lower timeframe
    williams_oversold_aligned = align_htf_to_ltf(prices, df_1d, williams_oversold)
    williams_overbought_aligned = align_htf_to_ltf(prices, df_1d, williams_overbought)
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_oversold_aligned[i]) or np.isnan(williams_overbought_aligned[i]) or
            np.isnan(ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: oversold + price above EMA
            if williams_oversold_aligned[i] and close[i] > ema_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: overbought + price below EMA
            elif williams_overbought_aligned[i] and close[i] < ema_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: oversold condition ends OR price crosses below EMA
            if (not williams_oversold_aligned[i]) or (close[i] < ema_20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: overbought condition ends OR price crosses above EMA
            if (not williams_overbought_aligned[i]) or (close[i] > ema_20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
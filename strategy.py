#!/usr/bin/env python3
# 4h_12h_Camarilla_R1_S1_Breakout_Volume_Momentum_v3
# Hypothesis: 4h Camarilla R1/S1 breakout with 12h momentum filter (price > 12h EMA20) and volume confirmation.
# Uses tighter volume filter (4x average) and longer EMA (40) to reduce trade frequency.
# Targets 15-25 trades/year to avoid fee drag. Works in bull/bear via momentum filter.

name = "4h_12h_Camarilla_R1_S1_Breakout_Volume_Momentum_v3"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    # Get 12h data for momentum filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA40 for momentum filter (slower = fewer signals)
    close_12h = df_12h['close'].values
    ema_40 = pd.Series(close_12h).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_aligned = align_htf_to_ltf(prices, df_12h, ema_40)
    
    # Calculate Camarilla levels (R1, S1) from previous day using 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 = close + 1.1*(high-low)/12
    # Camarilla S1 = close - 1.1*(high-low)/12
    r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align R1 and S1 to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 4h 10-period EMA for exit (slower exit = fewer reversals)
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Volume confirmation (4.0x 30-period average - much tighter)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 80
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_40_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_10[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Momentum filter from 12h EMA40
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        bullish_momentum = close_12h_aligned[i] > ema_40_aligned[i]
        bearish_momentum = close_12h_aligned[i] < ema_40_aligned[i]
        
        # Volume confirmation (4.0x average - much tighter)
        volume_surge = volume[i] > 4.0 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above R1 in bullish momentum with volume surge
            if close[i] > r1_aligned[i] and bullish_momentum and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S1 in bearish momentum with volume surge
            elif close[i] < s1_aligned[i] and bearish_momentum and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: price crosses below 10-period EMA
                if close[i] < ema_10[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price crosses above 10-period EMA
                if close[i] > ema_10[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
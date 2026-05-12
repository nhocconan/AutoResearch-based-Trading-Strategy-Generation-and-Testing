#!/usr/bin/env python3
# 1H_CAMARILLA_R1_S1_BREAKOUT_4HTREND_VOLUME_SPIKE
# Hypothesis: 1h price breaks of 4h Camarilla R1/S1 levels with volume spike and 4h trend filter.
# Uses 4h for signal direction/trend, 1h only for precise entry timing. Targets 15-30 trades/year.
# Works in bull/bear via trend filter and volume confirmation to avoid false breakouts.

name = "1H_CAMARILLA_R1_S1_BREAKOUT_4HTREND_VOLUME_SPIKE"
timeframe = "1h"
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
    
    # 4h timeframe for trend and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels: R1, S1
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla formulas
    R1 = close_4h + (high_4h - low_4h) * 1.0833 / 12
    S1 = close_4h - (high_4h - low_4h) * 1.0833 / 12
    
    # Align Camarilla levels to 1h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    
    # 4h EMA for trend filter
    ema4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema4h_aligned = align_htf_to_ltf(prices, df_4h, ema4h)
    
    # Volume spike detection (20-period volume MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > vol_ma * 2.0  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema4h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 with volume spike and 4h uptrend
            if close[i] > R1_aligned[i] and vol_spike[i] and close[i] > ema4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 with volume spike and 4h downtrend
            elif close[i] < S1_aligned[i] and vol_spike[i] and close[i] < ema4h_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to S1 level
            if close[i] < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price returns to R1 level
            if close[i] > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
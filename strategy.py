#!/usr/bin/env python3
# 12H_CAMARILLA_R1_S1_BREAKOUT_1D_TREND_VOLUME_SPIKE
# Hypothesis: On 12h timeframe, use daily Camarilla R1/S1 levels as breakout triggers.
# Enter long when price breaks above R1 with volume spike and daily uptrend (close > EMA34).
# Enter short when price breaks below S1 with volume spike and daily downtrend (close < EMA34).
# Exit when price returns to the opposite level (S1 for longs, R1 for shorts).
# Targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
# Uses 1d trend filter and volume confirmation to avoid false breakouts.
# Designed to work in both bull and bear markets via trend filter.

name = "12H_CAMARILLA_R1_S1_BREAKOUT_1D_TREND_VOLUME_SPIKE"
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
    
    # Camarilla levels from daily timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels: R1, S1
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas: R1 = C + (H-L)*1.1/2, S1 = C - (H-L)*1.1/2
    R1 = close_1d + (high_1d - low_1d) * 1.1 / 2
    S1 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 1d EMA for trend filter (34-period)
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    # Volume spike detection (20-period volume MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > vol_ma * 2.0  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 with volume spike and daily uptrend
            if close[i] > R1_aligned[i] and vol_spike[i] and close[i] > ema34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume spike and daily downtrend
            elif close[i] < S1_aligned[i] and vol_spike[i] and close[i] < ema34_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to S1 level
            if close[i] < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to R1 level
            if close[i] > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
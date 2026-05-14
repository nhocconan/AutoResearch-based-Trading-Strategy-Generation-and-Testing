#!/usr/bin/env python3
# 4H_CAMARILLA_R1_S1_BREAKOUT_12HTREND_VOLUME_SPIKE
# Hypothesis: Camarilla pivot levels (R1/S1) from 12h timeframe provide strong support/resistance.
# Enter long when price breaks above R1 with volume spike and 12h uptrend; short when price breaks below S1 with volume spike and 12h downtrend.
# Exit when price returns to the opposite Camarilla level (S1 for longs, R1 for shorts).
# Uses 12h trend filter to avoid counter-trend trades and volume spike to confirm breakout strength.
# Targets 25-40 trades/year to minimize fee drag while capturing meaningful breakouts.

name = "4H_CAMARILLA_R1_S1_BREAKOUT_12HTREND_VOLUME_SPIKE"
timeframe = "4h"
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
    
    # Camarilla levels from 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels: R1, S1
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla formulas
    R1 = close_12h + (high_12h - low_12h) * 1.0833 / 12
    S1 = close_12h - (high_12h - low_12h) * 1.0833 / 12
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_12h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_12h, S1)
    
    # 12h EMA for trend filter
    ema12 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema12_aligned = align_htf_to_ltf(prices, df_12h, ema12)
    
    # Volume spike detection (20-period volume MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > vol_ma * 2.0  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema12_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 with volume spike and 12h uptrend
            if close[i] > R1_aligned[i] and vol_spike[i] and close[i] > ema12_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume spike and 12h downtrend
            elif close[i] < S1_aligned[i] and vol_spike[i] and close[i] < ema12_aligned[i]:
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
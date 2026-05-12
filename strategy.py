#!/usr/bin/env python3
"""
4H_CAMARILLA_R1_S1_BREAKOUT_12HTREND_VOLUMESPIKE
Hypothesis: Camarilla R1/S1 levels derived from 12h timeframe act as strong support/resistance.
Breakouts above R1 (long) or below S1 (short) with 12h EMA50 trend filter and volume spike
provide high-probability entries. Works in bull markets (breakouts continue) and bear
markets (breakouts fail and reverse). Target: 20-50 trades/year on 4h timeframe.
"""
name = "4H_CAMARILLA_R1_S1_BREAKOUT_12HTREND_VOLUMESPIKE"
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
    
    # 12h data for Camarilla levels and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar: 
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    hl_range_12h = high_12h - low_12h
    camarilla_r1_12h = close_12h + 1.1 * hl_range_12h / 12
    camarilla_s1_12h = close_12h - 1.1 * hl_range_12h / 12
    
    # EMA50 for trend filter on 12h
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike: current 4h volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    # Align 12h indicators to 4h timeframe (wait for 12h bar to close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1_12h)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1_12h)
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least one prior bar for Camarilla levels
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R1 with volume spike in uptrend
            if (close[i] > camarilla_r1_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 with volume spike in downtrend
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks back below R1 or trend reversal
            if (close[i] < camarilla_r1_aligned[i] or 
                close[i] < ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks back above S1 or trend reversal
            if (close[i] > camarilla_s1_aligned[i] or 
                close[i] > ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
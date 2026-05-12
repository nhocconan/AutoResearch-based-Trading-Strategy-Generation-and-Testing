#!/usr/bin/env python3
"""
12H_WEEKLY_CAMARILLA_R1_S1_BREAKOUT_VOLUME
Hypothesis: Weekly Camarilla R1/S1 levels provide strong institutional support/resistance. Breakouts on 12h with volume confirmation and weekly trend filter capture major moves in both bull and bear markets. Weekly trend filter reduces false breakouts. Target: 15-25 trades/year.
"""
name = "12H_WEEKLY_CAMARILLA_R1_S1_BREAKOUT_VOLUME"
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
    
    # Weekly data for Camarilla levels (using previous week's range)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's range for Camarilla calculation
    ph = np.roll(high_1w, 1)  # previous week high
    pl = np.roll(low_1w, 1)   # previous week low
    pc = np.roll(close_1w, 1) # previous week close
    ph[0] = pl[0] = pc[0] = np.nan  # first week has no previous week
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    r1 = pc + (ph - pl) * 1.1 / 12
    s1 = pc - (ph - pl) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Weekly trend filter: EMA50 on weekly close
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike: current 12h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R1 with volume spike and price above weekly EMA50
            if (high[i] > r1_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 with volume spike and price below weekly EMA50
            elif (low[i] < s1_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below S1 or below weekly EMA50
            if (close[i] < s1_aligned[i] or 
                close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above R1 or above weekly EMA50
            if (close[i] > r1_aligned[i] or 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
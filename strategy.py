#!/usr/bin/env python3
"""
4H_CAMARILLA_R1_S1_BREAKOUT_12HTREND_VOLUME
Hypothesis: Camarilla pivot R1/S1 breakout with 12h EMA trend filter and volume confirmation.
Long when price breaks above R1 with 12h EMA uptrend and volume spike.
Short when price breaks below S1 with 12h EMA downtrend and volume spike.
Exit when price returns to pivot point (PP) or trend reverses.
Designed to capture institutional breakouts with trend alignment and volume validation.
Targets 25-40 trades/year to minimize fee drag with high-probability setups.
"""

name = "4H_CAMARILLA_R1_S1_BREAKOUT_12HTREND_VOLUME"
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
    
    # Calculate Camarilla pivot points from previous day
    # Using daily high/low/close from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    phigh = df_1d['high'].values
    plow = df_1d['low'].values
    pclose = df_1d['close'].values
    
    # Calculate pivot point and levels
    pp = (phigh + plow + pclose) / 3
    r1 = pp + (phigh - plow) * 1.1 / 12
    s1 = pp - (phigh - plow) * 1.1 / 12
    
    # Align to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 12h EMA for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    pclose_12h = df_12h['close'].values
    ema12h = pd.Series(pclose_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema12h_aligned = align_htf_to_ltf(prices, df_12h, ema12h)
    
    # Volume spike detection (20-period volume MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x volume MA
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            # LONG: price breaks above R1 with uptrend and volume spike
            if close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1] and \
               ema12h_aligned[i] > ema12h_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1 with downtrend and volume spike
            elif close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1] and \
                 ema12h_aligned[i] < ema12h_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to pivot point or trend reverses
            if close[i] < pp_aligned[i] or ema12h_aligned[i] < ema12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to pivot point or trend reverses
            if close[i] > pp_aligned[i] or ema12h_aligned[i] > ema12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
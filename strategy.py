#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_Volume_Confirmation_v2
Hypothesis: Camarilla pivot R1/S1 breakouts with volume confirmation and 12h EMA trend filter capture institutional order flow while avoiding false breakouts. Designed for ~30 trades/year to minimize fee drift and work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day (1d timeframe)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla calculation: R1/S1 from previous day
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    camarilla_range = high_1d - low_1d
    r1_1d = close_1d + (1.1 * camarilla_range) / 12
    s1_1d = close_1d - (1.1 * camarilla_range) / 12
    
    # Align to 4h timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 12h EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_12h_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        ema_12h_val = ema_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and above 12h EMA
            if price > r1 and vol_spike and price > ema_12h_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and below 12h EMA
            elif price < s1 and vol_spike and price < ema_12h_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price below S1 or below 12h EMA
            if price < s1 or price < ema_12h_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price above R1 or above 12h EMA
            if price > r1 or price > ema_12h_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_Confirmation_v2"
timeframe = "4h"
leverage = 1.0
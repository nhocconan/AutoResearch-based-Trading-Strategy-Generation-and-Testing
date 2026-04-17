#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_MultiTF_v1
Hypothesis: Camarilla pivot levels (R1/S1) from 1d act as strong support/resistance. 
Price breaking above R1 with volume spike and 1w trend bias (price > 1w EMA200) triggers long.
Price breaking below S1 with volume spike and 1w trend bias (price < 1w EMA200) triggers short.
Exit when price returns to the 1d pivot (central level) or reverses with volume confirmation.
Designed to work in both bull (breakouts continue) and bear (mean reversion at extremes) markets.
Target: 20-50 trades/year (~80-200 total over 4 years) to avoid fee drag.
"""

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
    
    # === 1d Camarilla Pivot Levels ===
    df_1d = get_htf_data(prices, '1d')
    # Typical price for pivot calculation
    typical_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    # Calculate pivot and support/resistance levels
    pivot_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    range_1d = df_1d['high'] - df_1d['low']
    # Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    r1_1d = df_1d['close'] + range_1d * 1.1 / 12
    s1_1d = df_1d['close'] - range_1d * 1.1 / 12
    # Align to 4h timeframe (wait for 1d bar to close)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d.values)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d.values)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d.values)
    
    # === 1w EMA200 for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # === Volume Spike (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1, volume spike, price above 1w EMA200 (bullish bias)
            if (close[i] > r1_1d_aligned[i] and 
                vol_spike[i] and 
                close[i] > ema_200_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1, volume spike, price below 1w EMA200 (bearish bias)
            elif (close[i] < s1_1d_aligned[i] and 
                  vol_spike[i] and 
                  close[i] < ema_200_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price returns to pivot OR closes below S1 with volume (reversal)
            if (close[i] <= pivot_1d_aligned[i] or 
                (close[i] < s1_1d_aligned[i] and vol_spike[i])):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot OR closes above R1 with volume (reversal)
            if (close[i] >= pivot_1d_aligned[i] or 
                (close[i] > r1_1d_aligned[i] and vol_spike[i])):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_MultiTF_v1"
timeframe = "4h"
leverage = 1.0
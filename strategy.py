#!/usr/bin/env python3
# 12h_1w_1d_Pivot_R1_S1_Breakout_VolumeFilter
# Hypothesis: Weekly and daily pivot levels (R1/S1) act as strong support/resistance in BTC/ETH.
# Breakouts with volume confirmation capture sustained moves in both bull and bear markets.
# Using 12h timeframe reduces trade frequency; weekly trend filter avoids counter-trend trades.

name = "12h_1w_1d_Pivot_R1_S1_Breakout_VolumeFilter"
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
    
    # Get weekly and daily data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 1 or len(df_1d) < 2:
        return np.zeros(n)
    
    # Weekly trend filter: price above/below weekly VWAP
    # VWAP = sum(price * volume) / sum(volume)
    vwap_num = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3 * df_1w['volume']
    vwap_den = df_1w['volume']
    weekly_vwap = (vwap_num.cumsum() / vwap_den.cumsum()).values
    weekly_vwap_aligned = align_htf_to_ltf(prices, df_1w, weekly_vwap)
    
    # Daily pivot and Camarilla levels (R1/S1)
    ph = df_1d['high'].values
    pl = df_1d['low'].values
    pc = df_1d['close'].values
    
    # Pivot point
    p = (ph + pl + pc) / 3
    # Camarilla R1 and S1
    r1 = p + (ph - pl) * 1.1 / 12
    s1 = p - (ph - pl) * 1.1 / 12
    
    # Align to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(volume_filter[i]) or np.isnan(weekly_vwap_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly VWAP (uptrend) + breaks above R1 + volume
            if close[i] > weekly_vwap_aligned[i] and close[i] > r1_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly VWAP (downtrend) + breaks below S1 + volume
            elif close[i] < weekly_vwap_aligned[i] and close[i] < s1_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1 (reversal) or falls below weekly VWAP
            if close[i] < s1_aligned[i] or close[i] < weekly_vwap_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above R1 (reversal) or rises above weekly VWAP
            if close[i] > r1_aligned[i] or close[i] > weekly_vwap_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
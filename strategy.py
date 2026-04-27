#!/usr/bin/env python3
"""
4h_PriceAction_Reversal_at_12h_Pivot_Trend
Hypothesis: Price reverses at 12-hour pivot points (PP, R1, S1) when confirmed by 12h EMA50 trend and volume spikes. 
Uses pivot rejection for mean reversion in ranging markets and breakout continuation in trending markets.
Works in bull/bear via trend filter. Target: 20-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for pivot calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 12h pivot points from previous 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pivot Point (PP), Resistance 1 (R1), Support 1 (S1)
    PP = (high_12h + low_12h + close_12h) / 3.0
    R1 = 2 * PP - low_12h
    S1 = 2 * PP - high_12h
    
    # Align to 4h timeframe (previous 12h bar's pivots available at open)
    PP_aligned = align_htf_to_ltf(prices, df_12h, PP)
    R1_aligned = align_htf_to_ltf(prices, df_12h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_12h, S1)
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 20  # need 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(PP_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price bounces off S1 with volume spike and uptrend on 12h
            if (close[i] <= S1_aligned[i] * 1.005 and close[i] >= S1_aligned[i] * 0.995 and 
                volume_spike[i] and close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price rejects R1 with volume spike and downtrend on 12h
            elif (close[i] >= R1_aligned[i] * 0.995 and close[i] <= R1_aligned[i] * 1.005 and 
                  volume_spike[i] and close[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price reaches PP or trend fails
            if (close[i] >= PP_aligned[i] or 
                close[i] < ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches PP or trend fails
            if (close[i] <= PP_aligned[i] or 
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_PriceAction_Reversal_at_12h_Pivot_Trend"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
"""
6h_PriceAction_Reversal_at_DailyPivots
Hypothesis: Price reversals at daily pivot points (PP, R1, S1) with volume exhaustion and 1d trend filter capture mean-reversion moves in ranging markets and pullbacks in trends. Works in both bull (sell at resistance) and bear (buy at support) markets. Target: 15-30 trades/year per symbol.
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
    
    # Get 1d data for pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Daily pivot points from previous day
    high_prev = df_1d['high'].values
    low_prev = df_1d['low'].values
    close_prev = df_1d['close'].values
    
    # Pivot Point (PP), Resistance 1 (R1), Support 1 (S1)
    PP = (high_prev + low_prev + close_prev) / 3
    R1 = 2 * PP - low_prev
    S1 = 2 * PP - high_prev
    
    # Align to 6h timeframe (previous day's pivots available at open)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume exhaustion: current volume < 50% of 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_exhaustion = volume < (vol_ma * 0.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 20  # need 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(PP_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price at or below S1 with volume exhaustion and above 1d EMA50 (buy pullback in uptrend)
            if (close[i] <= S1_aligned[i] and volume_exhaustion[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price at or above R1 with volume exhaustion and below 1d EMA50 (sell rally in downtrend)
            elif (close[i] >= R1_aligned[i] and volume_exhaustion[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price reaches PP or trend fails
            if (close[i] >= PP_aligned[i] or 
                close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches PP or trend fails
            if (close[i] <= PP_aligned[i] or 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_PriceAction_Reversal_at_DailyPivots"
timeframe = "6h"
leverage = 1.0
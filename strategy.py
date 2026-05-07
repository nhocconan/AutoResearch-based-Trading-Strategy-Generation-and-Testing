#!/usr/bin/env python3
# 6H_WeeklyPivot_Momentum_Volume
# Hypothesis: Combines weekly pivot points (calculated from prior week) with 60-period EMA trend filter and volume confirmation.
# Long when price breaks above weekly R1 in uptrend (close > EMA60) with volume spike.
# Short when price breaks below weekly S1 in downtrend (close < EMA60) with volume spike.
# Exits when price returns to weekly pivot (PP) level.
# Designed for low trade frequency (<30/year) and works in both bull and bear markets by following weekly trend.

name = "6H_WeeklyPivot_Momentum_Volume"
timeframe = "6h"
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
    
    # Get weekly data for pivot calculation
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    # Calculate weekly OHLC for pivot points (using previous week's data)
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Weekly pivot points: PP = (H + L + C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    pp_w = (high_w + low_w + close_w) / 3
    r1_w = 2 * pp_w - low_w
    s1_w = 2 * pp_w - high_w
    
    # 60-period EMA for trend filter (on 6h timeframe)
    ema60 = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Align weekly pivot levels to 6h timeframe
    pp_w_aligned = align_htf_to_ltf(prices, df_w, pp_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_w, s1_w)
    
    # Volume filter: current volume > 2.0x average volume (24-period ~ 4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure we have EMA and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(pp_w_aligned[i]) or np.isnan(r1_w_aligned[i]) or 
            np.isnan(s1_w_aligned[i]) or np.isnan(ema60[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above R1 + Uptrend (close > EMA60) + volume spike
            if (close[i] > r1_w_aligned[i] and 
                close[i] > ema60[i] and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 + Downtrend (close < EMA60) + volume spike
            elif (close[i] < s1_w_aligned[i] and 
                  close[i] < ema60[i] and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price returns to weekly pivot (PP) level
            if close[i] <= pp_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price returns to weekly pivot (PP) level
            if close[i] >= pp_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
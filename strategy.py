#!/usr/bin/env python3
# 4h_TRIX_ZeroCross_VolumeTrend
# Hypothesis: TRIX (Triple Exponential Average) zero-cross signals with volume confirmation
# and 1d EMA34 trend filter capture momentum shifts while avoiding false signals in both bull and bear markets.
# TRIX is less noisy than MACD and better at identifying trend changes. Volume filter ensures
# breakouts have institutional participation. Target: 20-50 trades per year (~80-200 over 4 years) with position size 0.25.

name = "4h_TRIX_ZeroCross_VolumeTrend"
timeframe = "4h"
leverage = 1.0

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
    
    # Load 1d data ONCE for TRIX calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    # Calculate TRIX on daily close
    # TRIX = EMA(EMA(EMA(close, period), period), period) - previous value
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = ema3 - np.roll(ema3, 1)  # Difference from previous value
    trix[0] = 0  # First value has no previous
    
    # Align TRIX to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 periods for volume MA and TRIX stabilization
    
    for i in range(start_idx, n):
        if (np.isnan(trix_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # TRIX zero-cross signals
        trix_cross_up = trix_aligned[i] > 0 and trix_aligned[i-1] <= 0
        trix_cross_down = trix_aligned[i] < 0 and trix_aligned[i-1] >= 0
        
        # Volume confirmation: volume > 1.3x average
        volume_confirm = vol_ratio[i] > 1.3
        
        # Trend filter from 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: TRIX crosses above zero + volume + uptrend
            if trix_cross_up and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero + volume + downtrend
            elif trix_cross_down and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX crosses back below zero (momentum loss) or trend reversal
            if trix_aligned[i] < 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses back above zero (momentum loss) or trend reversal
            if trix_aligned[i] > 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
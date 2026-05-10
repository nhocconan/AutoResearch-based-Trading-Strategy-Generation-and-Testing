#!/usr/bin/env python3
# 4h_Chaikin_Oscillator_Trend_Confirmation
# Hypothesis: Chaikin Oscillator (3,10) crossing zero with daily trend confirmation and volume spike filters captures momentum shifts in both bull and bear markets. Daily trend avoids counter-trend trades, volume reduces false signals. Designed for low frequency (~20-50 trades/year) to minimize fee drift.

name = "4h_Chaikin_Oscillator_Trend_Confirmation"
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
    
    # Money Flow Multiplier and Volume
    mfm = np.where((high - low) != 0, ((close - low) - (high - close)) / (high - low), 0)
    mfv = mfm * volume
    
    # Chaikin Oscillator: (3-period EMA of MFV) - (10-period EMA of MFV)
    mfv_series = pd.Series(mfv)
    ema3 = mfv_series.ewm(span=3, adjust=False, min_periods=3).mean().values
    ema10 = mfv_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    chaikin = ema3 - ema10
    
    # Daily trend filter: EMA50 on daily close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema50_1d
    trend_1d_down = close_1d < ema50_1d
    
    # Align daily trend to 4h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Volume confirmation: 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(chaikin[i]) or 
            np.isnan(trend_1d_up_aligned[i]) or 
            np.isnan(trend_1d_down_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Enter long: Chaikin crosses above zero with daily uptrend and volume
            if chaikin[i] > 0 and chaikin[i-1] <= 0 and trend_1d_up_aligned[i] > 0.5 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: Chaikin crosses below zero with daily downtrend and volume
            elif chaikin[i] < 0 and chaikin[i-1] >= 0 and trend_1d_down_aligned[i] > 0.5 and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when Chaikin crosses below zero or trend fails
            if chaikin[i] < 0 and chaikin[i-1] >= 0 or trend_1d_up_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when Chaikin crosses above zero or trend fails
            if chaikin[i] > 0 and chaikin[i-1] <= 0 or trend_1d_down_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
"""
4h_TRIX_Volume_Spike_1dTrend_HTF
Hypothesis: TRIX (triple smoothed EMA) captures momentum with reduced noise. 
Long when TRIX crosses above zero with 1-day uptrend and volume spike.
Short when TRIX crosses below zero with 1-day downtrend and volume spike.
Volume spike (>2x 20-period average) confirms momentum. 
Designed to work in both bull and bear markets by following the 1-day trend.
Target: ~20-40 trades/year to minimize fee drag.
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate TRIX: triple EMA of ROC
    # ROC = (close - close[n]) / close[n] * 100
    roc = np.zeros_like(close)
    roc[1:] = (close[1:] - close[:-1]) / close[:-1] * 100
    
    # Triple EMA of ROC
    ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = ema3  # TRIX is the final smoothed EMA
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Volume confirmation: >2x 20-period MA (stricter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(trix[i]) or 
            np.isnan(trix_signal[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1-day EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # TRIX momentum signals
        trix_cross_up = trix[i] > trix_signal[i] and trix[i-1] <= trix_signal[i-1]
        trix_cross_down = trix[i] < trix_signal[i] and trix[i-1] >= trix_signal[i-1]
        
        # Volume confirmation (>2x average for stronger signal)
        vol_confirm = volume[i] > (2.0 * vol_ma_20[i])
        
        # Entry logic: TRIX cross in direction of trend with volume
        long_entry = vol_confirm and uptrend and trix_cross_up
        short_entry = vol_confirm and downtrend and trix_cross_down
        
        # Exit logic: opposite TRIX cross or trend change
        long_exit = trix_cross_down or (not uptrend)
        short_exit = trix_cross_up or (not downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_TRIX_Volume_Spike_1dTrend_HTF"
timeframe = "4h"
leverage = 1.0
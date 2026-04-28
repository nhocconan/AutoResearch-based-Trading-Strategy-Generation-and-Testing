#!/usr/bin/env python3
"""
4h_TRIX_Volume_Spike_1dTrend_HTF
Hypothesis: TRIX (triple smoothed EMA) crosses signal line with volume spike confirmation and 1d EMA trend filter. Works in bull markets (long when TRIX crosses above signal) and bear markets (short when TRIX crosses below signal). Target: 20-30 trades/year to minimize fee drag while capturing momentum shifts.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # TRIX calculation (15-period triple EMA)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = pd.Series(ema3).pct_change(1) * 100  # Percentage change
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Volume confirmation: >1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or 
            np.isnan(trix_signal[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # TRIX signals
        trix_cross_above = trix[i] > trix_signal[i] and trix[i-1] <= trix_signal[i-1]
        trix_cross_below = trix[i] < trix_signal[i] and trix[i-1] >= trix_signal[i-1]
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Entry logic: TRIX cross in direction of trend with volume
        long_entry = vol_confirm and uptrend and trix_cross_above
        short_entry = vol_confirm and downtrend and trix_cross_below
        
        # Exit logic: opposite TRIX cross or trend change
        long_exit = trix_cross_below or (not uptrend)
        short_exit = trix_cross_above or (not downtrend)
        
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
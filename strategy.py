#!/usr/bin/env python3
"""
4h_TRIX_14_ZeroCross_12hTrend_VolumeFilter
Hypothesis: Use TRIX(14) zero crossovers as momentum signals, filtered by 12h EMA trend and volume spike (>1.5x average). TRIX captures momentum reversals early, while 12h trend ensures directional alignment and volume filter avoids false breakouts. Designed for fewer trades (~20-30/year) with clear entry/exit rules. Works in bull/bear by requiring trend alignment.
"""

name = "4h_TRIX_14_ZeroCross_12hTrend_VolumeFilter"
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
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    daily_close = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate TRIX(14) on close
    # TRIX = EMA(EMA(EMA(close, 14), 14), 14) then % change
    ema1 = pd.Series(close).ewm(span=14, adjust=False, min_periods=14).mean().values
    ema2 = pd.Series(ema1).ewm(span=14, adjust=False, min_periods=14).mean().values
    ema3 = pd.Series(ema2).ewm(span=14, adjust=False, min_periods=14).mean().values
    trix = np.diff(ema3, prepend=ema3[0]) / ema3 * 100
    
    # Volume confirmation: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(trix[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 12h trend using aligned close
        daily_close_aligned = align_htf_to_ltf(prices, df_12h, daily_close)
        if np.isnan(daily_close_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        trend_up = daily_close_aligned[i] > ema_50_12h_aligned[i]
        trend_down = daily_close_aligned[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: TRIX crosses above zero with uptrend and volume spike
            if trix[i] > 0 and trix[i-1] <= 0 and trend_up and vol_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with downtrend and volume spike
            elif trix[i] < 0 and trix[i-1] >= 0 and trend_down and vol_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX crosses below zero or trend turns down
            if trix[i] < 0 and trix[i-1] >= 0 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX crosses above zero or trend turns up
            if trix[i] > 0 and trix[i-1] <= 0 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
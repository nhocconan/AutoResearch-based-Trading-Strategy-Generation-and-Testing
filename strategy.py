#!/usr/bin/env python3
"""
1d_1w_Multiplier_Momentum_Trend
Hypothesis: Use weekly momentum to determine trend direction, then trade on daily price pullbacks to EMA with volume confirmation. Works in bull markets (buy dips) and bear markets (sell rallies) by following the weekly trend. Targets 10-25 trades per year with strict entry conditions.
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
    
    # Get daily data for EMA and volume
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA(34) for pullback entries
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get weekly data for trend direction
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly momentum: ROC(4) - rate of change over 4 weeks
    roc_1w = np.full_like(close_1w, np.nan)
    for i in range(4, len(close_1w)):
        roc_1w[i] = (close_1w[i] - close_1w[i-4]) / close_1w[i-4] * 100
    
    # Align all indicators to daily timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    roc_1w_aligned = align_htf_to_ltf(prices, df_1w, roc_1w)
    
    # Volume confirmation: current volume > 1.5 x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need EMA(34) and enough data for ROC
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(roc_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: weekly uptrend (positive momentum) + price pulls back to EMA + volume
            if (roc_1w_aligned[i] > 0 and  # Weekly uptrend
                close[i] <= ema_1d_aligned[i] * 1.02 and  # Near EMA (within 2%)
                close[i] >= ema_1d_aligned[i] * 0.98 and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: weekly downtrend (negative momentum) + price rallies to EMA + volume
            elif (roc_1w_aligned[i] < 0 and  # Weekly downtrend
                  close[i] >= ema_1d_aligned[i] * 0.98 and  # Near EMA (within 2%)
                  close[i] <= ema_1d_aligned[i] * 1.02 and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: weekly momentum turns negative or price moves significantly above EMA
            if (roc_1w_aligned[i] < 0 or 
                close[i] > ema_1d_aligned[i] * 1.05):  # 5% above EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: weekly momentum turns positive or price moves significantly below EMA
            if (roc_1w_aligned[i] > 0 or 
                close[i] < ema_1d_aligned[i] * 0.95):  # 5% below EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Multiplier_Momentum_Trend"
timeframe = "1d"
leverage = 1.0
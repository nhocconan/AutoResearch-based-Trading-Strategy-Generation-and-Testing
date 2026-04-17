#!/usr/bin/env python3
"""
6h_ElderRay_RayForce_V1
6-hour strategy using Elder Ray power (Bull/Bear) with 1-week force index filter.
Targets trend exhaustion points in both bull and bear markets via divergence between
price and force. Low-frequency design aims for 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Weekly Force Index (13-period EMA of price change * volume) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Price change
    price_change_1w = np.diff(close_1w, prepend=close_1w[0])
    # Force Index = price change * volume
    force_1w = price_change_1w * volume_1w
    # EMA of Force Index (13-period)
    ema_force_1w = pd.Series(force_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Align weekly force to 6h
    force_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_force_1w)
    
    # === Daily Elder Ray Power (13-period EMA) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 13-period EMA
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power_1d = high_1d - ema_13_1d
    # Bear Power = Low - EMA13
    bear_power_1d = low_1d - ema_13_1d
    
    # Align to 6h
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # === 6h EMA for trend context ===
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(force_1w_aligned[i]) or 
            np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i]) or 
            np.isnan(ema_50[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Bull Power rising AND weekly Force turning up from negative
            if (bull_power_1d_aligned[i] > bull_power_1d_aligned[i-1] and 
                force_1w_aligned[i] > 0 and 
                force_1w_aligned[i-1] <= 0 and
                close[i] > ema_50[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Bear Power falling (more negative) AND weekly Force turning down from positive
            elif (bear_power_1d_aligned[i] < bear_power_1d_aligned[i-1] and 
                  force_1w_aligned[i] < 0 and 
                  force_1w_aligned[i-1] >= 0 and
                  close[i] < ema_50[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: power divergence or force reversal
        elif position == 1:
            # Exit long: Bear Power rising (less negative) OR weekly Force turning down
            if (bear_power_1d_aligned[i] > bear_power_1d_aligned[i-1] or
                force_1w_aligned[i] < 0 and force_1w_aligned[i-1] >= 0):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bull Power falling OR weekly Force turning up
            if (bull_power_1d_aligned[i] < bull_power_1d_aligned[i-1] or
                force_1w_aligned[i] > 0 and force_1w_aligned[i-1] <= 0):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_RayForce_V1"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
"""
6h_ElderRay_ZeroCross_1dTrend_VolumeFilter
Hypothesis: 6h Elder Ray (Bull Power/Bear Power) zero cross with 1d EMA50 trend filter and volume confirmation.
Long when Bull Power crosses above zero with 1d EMA50 uptrend and volume > 2.0x 20-period average.
Short when Bear Power crosses below zero with 1d EMA50 downtrend and volume > 2.0x 20-period average.
Exit on opposite zero cross or trend reversal.
Uses discrete sizing (0.25) to minimize fee churn. Target: 12-30 trades/year.
Works in bull via trend-following momentum, in bear via mean reversion at extreme power levels.
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
    
    # Get 6h data for Elder Ray calculations (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate 13-period EMA for 6h close
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_6h = high_6h - ema_13_6h
    bear_power_6h = low_6h - ema_13_6h
    
    # Align Elder Ray to original timeframe
    bull_power_6h_aligned = align_htf_to_ltf(prices, df_6h, bull_power_6h)
    bear_power_6h_aligned = align_htf_to_ltf(prices, df_6h, bear_power_6h)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA50 for trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_6h_aligned[i]) or np.isnan(bear_power_6h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: Bull Power crosses above zero with uptrend and volume spike
            bull_cross_up = (bull_power_6h_aligned[i] > 0) and (bull_power_6h_aligned[i-1] <= 0)
            # Short: Bear Power crosses below zero with downtrend and volume spike
            bear_cross_down = (bear_power_6h_aligned[i] < 0) and (bear_power_6h_aligned[i-1] >= 0)
            
            long_signal = bull_cross_up and (close[i] > ema_50_1d_aligned[i]) and vol_spike[i]
            short_signal = bear_cross_down and (close[i] < ema_50_1d_aligned[i]) and vol_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: Bear Power crosses below zero or trend reverses
            bear_cross_down = (bear_power_6h_aligned[i] < 0) and (bear_power_6h_aligned[i-1] >= 0)
            exit_signal = bear_cross_down or (close[i] < ema_50_1d_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: Bull Power crosses above zero or trend reverses
            bull_cross_up = (bull_power_6h_aligned[i] > 0) and (bull_power_6h_aligned[i-1] <= 0)
            exit_signal = bull_cross_up or (close[i] > ema_50_1d_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_ZeroCross_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0
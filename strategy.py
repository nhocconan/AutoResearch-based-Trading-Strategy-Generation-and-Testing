#!/usr/bin/env python3
"""
6h_ElderRay_ZeroCross_1dTrend_VolumeFilter
Hypothesis: Elder Ray Bull/Bear Power zero-cross signals filtered by 1d EMA50 trend and volume spikes.
Long when Bull Power crosses above zero in uptrend (close > 1d EMA50) with volume confirmation.
Short when Bear Power crosses below zero in downtrend (close < 1d EMA50) with volume confirmation.
Exit when power reverses sign or trend fails. Designed for low frequency and robustness in bull/bear markets.
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
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_6h - ema_13_6h
    bear_power = low_6h - ema_13_6h
    
    # Align Elder Ray components to original timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        bull = bull_power_aligned[i]
        bear = bear_power_aligned[i]
        prev_bull = bull_power_aligned[i-1] if i > 0 else 0.0
        prev_bear = bear_power_aligned[i-1] if i > 0 else 0.0
        
        if position == 0:
            # Regime-based entry logic
            if close[i] > ema_trend:  # Uptrend regime (daily)
                # Long: Bull Power crosses above zero with volume spike
                long_signal = (bull > 0) and (prev_bull <= 0) and vol_spike[i]
            else:  # Downtrend regime (daily)
                # Short: Bear Power crosses below zero with volume spike
                short_signal = (bear < 0) and (prev_bear >= 0) and vol_spike[i]
            
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
            # Exit conditions: Bull Power crosses below zero or trend failure
            exit_signal = (bull < 0) or (close[i] < ema_trend * 0.99)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: Bear Power crosses above zero or trend failure
            exit_signal = (bear > 0) or (close[i] > ema_trend * 1.01)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_ZeroCross_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
"""
4h_Chaikin_Money_Flow_Crossover_12hTrend_v1
Hypothesis: Chaikin Money Flow (CMF) crossing above/below zero, confirmed by 12h EMA trend and volume spikes, captures institutional flow-driven moves. Works in bull/bear as CMF reflects buying/selling pressure regardless of price direction. Targets 20-50 trades/year via trend and volume filters to reduce false signals and minimize fee drag.
"""
name = "4h_Chaikin_Money_Flow_Crossover_12hTrend_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA20 for trend filter
    ema_20_12h = pd.Series(df_12h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Chaikin Money Flow (20-period)
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Money Flow Volume = Money Flow Multiplier * Volume
    # CMF = Sum(Money Flow Volume, 20) / Sum(Volume, 20)
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = np.where((high - low) == 0, 0, mfm)  # avoid division by zero
    mfv = mfm * volume
    
    # Sum of MFV and Volume over 20 periods
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    vol_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    cmf = mfv_sum / vol_sum
    cmf = np.where(vol_sum == 0, 0, cmf)  # avoid division by zero
    
    # Volume filter: current volume > 1.3 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(20, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(ema_20_12h_aligned[i]) or np.isnan(cmf[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 6 bars between trades (24 hours on 4h TF) to reduce frequency
            if bars_since_exit < 6:
                continue
                
            # Long: CMF crosses above zero + 12h uptrend + volume filter
            if (cmf[i] > 0 and cmf[i-1] <= 0 and 
                close[i] > ema_20_12h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: CMF crosses below zero + 12h downtrend + volume filter
            elif (cmf[i] < 0 and cmf[i-1] >= 0 and 
                  close[i] < ema_20_12h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: CMF crosses back through zero (mean reversion of flow)
            if (position == 1 and cmf[i] < 0) or (position == -1 and cmf[i] > 0):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals
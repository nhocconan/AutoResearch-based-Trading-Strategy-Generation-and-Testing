#!/usr/bin/env python3
name = "6h_ElderRay_BullPower_1dTrend"
timeframe = "6h"
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
    
    # 1d trend filter: 34 EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Elder Ray: Bull Power and Bear Power (13-period EMA)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume filter: 20-period average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08:00 - 20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # Ensure volume and EMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN or invalid
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(ema_13[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period average
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 (buying pressure) in 1d uptrend, with volume spike
            if (bull_power[i] > 0 and 
                close[i] > ema_34_1d_aligned[i] and   # 1d uptrend filter
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (selling pressure) in 1d downtrend, with volume spike
            elif (bear_power[i] < 0 and 
                  close[i] < ema_34_1d_aligned[i] and   # 1d downtrend filter
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Elder Power crosses zero (loss of momentum)
            if position == 1 and bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            elif position == -1 and bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals
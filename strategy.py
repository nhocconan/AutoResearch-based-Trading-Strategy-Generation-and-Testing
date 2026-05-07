#!/usr/bin/env python3
name = "6h_FisherTransform_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Fisher Transform on 6h close prices (period=9)
    # Calculate median price
    median_price = (high + low) / 2
    
    # Normalize to [-1, 1] range over lookback period
    lookback = 9
    highest = pd.Series(median_price).rolling(window=lookback, min_periods=lookback).max().values
    lowest = pd.Series(median_price).rolling(window=lookback, min_periods=lookback).min().values
    
    # Avoid division by zero
    range_val = highest - lowest
    range_val = np.where(range_val == 0, 1e-10, range_val)
    
    # Normalize and bound to [-0.999, 0.999] to avoid infinity
    value = 2 * ((median_price - lowest) / range_val - 0.5)
    value = np.clip(value, -0.999, 0.999)
    
    # Fisher Transform formula: 0.5 * ln((1+value)/(1-value))
    fisher = 0.5 * np.log((1 + value) / (1 - value))
    
    # Smooth with 3-period EMA
    fisher_smooth = pd.Series(fisher).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Volume filter: current volume > 1.8 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_ok = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback + 2, 31)  # Fisher needs lookback + smoothing, volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(fisher_smooth[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Fisher crosses above -1.5 + above daily EMA34 + volume spike
            if fisher_smooth[i] > -1.5 and fisher_smooth[i-1] <= -1.5 and close[i] > ema_34_1d_aligned[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: Fisher crosses below +1.5 + below daily EMA34 + volume spike
            elif fisher_smooth[i] < 1.5 and fisher_smooth[i-1] >= 1.5 and close[i] < ema_34_1d_aligned[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Fisher crosses zero or trend reverses
            if position == 1:
                if fisher_smooth[i] < 0 or close[i] < ema_34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if fisher_smooth[i] > 0 or close[i] > ema_34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
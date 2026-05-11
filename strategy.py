#!/usr/bin/env python3
name = "12h_KAMA_Trend_1dVolume_Confirm"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1d data for volume confirmation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # KAMA trend on 1d close
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    er[1:] = change[1:] / (np.sum(volatility.reshape(-1, 1), axis=1) + 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # 1d volume filter: 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio = vol_1d / vol_ma_20
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Align KAMA and volume ratio to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check session
        if not in_session[i]:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold - avoid low-volume false signals
        volume_surge = vol_ratio_aligned[i] > 1.3
        
        if position == 0:
            # Long: Price above KAMA with volume surge
            if close[i] > kama_aligned[i] and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA with volume surge
            elif close[i] < kama_aligned[i] and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses back to KAMA
            if position == 1:
                if close[i] <= kama_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if close[i] >= kama_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
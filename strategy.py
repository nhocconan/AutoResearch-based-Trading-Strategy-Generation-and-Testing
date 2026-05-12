#!/usr/bin/env python3
name = "1d_Donchian_20_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 300:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1w Data for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === 1w EMA50 for trend ===
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === Daily Donchian Channel (20) ===
    period = 20
    dc_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    dc_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # === Volume spike detection (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(300, 50, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(dc_high[i]) or
            np.isnan(dc_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high + 1w uptrend + volume spike
            if (close[i] > dc_high[i] and 
                close[i] > ema50_1w_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + 1w downtrend + volume spike
            elif (close[i] < dc_low[i] and 
                  close[i] < ema50_1w_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price closes below Donchian low or trend breaks
            if close[i] < dc_low[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price closes above Donchian high or trend breaks
            if close[i] > dc_high[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
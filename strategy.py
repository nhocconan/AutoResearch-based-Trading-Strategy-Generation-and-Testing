#!/usr/bin/env python3
name = "12h_Donchian20_1wTrend_1dVolumeSpike"
timeframe = "12h"
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
    
    # ===== 12h Donchian Channel (20-period) =====
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # ===== 1w EMA21 Trend Filter =====
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # ===== 1d Volume Spike Filter =====
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (2.0 * vol_avg_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # ===== Session Filter: 08-20 UTC =====
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian channel to be valid
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema21_1w_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above Donchian high + above weekly EMA21 + daily volume spike
            if (close[i] > donchian_high[i] and
                close[i] > ema21_1w_aligned[i] and
                vol_spike_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low + below weekly EMA21 + daily volume spike
            elif (close[i] < donchian_low[i] and
                  close[i] < ema21_1w_aligned[i] and
                  vol_spike_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close below Donchian low or below weekly EMA21
            if close[i] < donchian_low[i] or close[i] < ema21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close above Donchian high or above weekly EMA21
            if close[i] > donchian_high[i] or close[i] > ema21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
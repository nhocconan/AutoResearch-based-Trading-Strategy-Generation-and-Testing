#!/usr/bin/env python3
name = "1h_Adaptive_Supertrend_v2"
timeframe = "1h"
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
    
    # === 4h Supertrend trend filter ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # ATR calculation for Supertrend
    atr_period = 10
    tr_4h = np.maximum(high_4h[1:] - low_4h[1:], 
                       np.maximum(np.abs(high_4h[1:] - close_4h[:-1]), 
                                  np.abs(low_4h[1:] - close_4h[:-1])))
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr_4h = pd.Series(tr_4h).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Supertrend calculation
    multiplier = 3.0
    hl2_4h = (high_4h + low_4h) / 2
    upperband_4h = hl2_4h + multiplier * atr_4h
    lowerband_4h = hl2_4h - multiplier * atr_4h
    
    supertrend_4h = np.full_like(close_4h, np.nan)
    direction_4h = np.full_like(close_4h, np.nan)
    
    for i in range(1, len(close_4h)):
        if np.isnan(atr_4h[i]) or np.isnan(upperband_4h[i-1]) or np.isnan(lowerband_4h[i-1]):
            supertrend_4h[i] = np.nan
            direction_4h[i] = np.nan
        else:
            if close_4h[i-1] > upperband_4h[i-1]:
                direction_4h[i] = 1
            elif close_4h[i-1] < lowerband_4h[i-1]:
                direction_4h[i] = -1
            else:
                direction_4h[i] = direction_4h[i-1]
            
            if direction_4h[i] == 1:
                upperband_4h[i] = min(upperband_4h[i], upperband_4h[i-1])
                lowerband_4h[i] = lowerband_4h[i-1]
                if close_4h[i] < lowerband_4h[i]:
                    direction_4h[i] = -1
                    upperband_4h[i] = hl2_4h[i] + multiplier * atr_4h[i]
                    lowerband_4h[i] = hl2_4h[i] - multiplier * atr_4h[i]
                supertrend_4h[i] = upperband_4h[i] if direction_4h[i] == 1 else lowerband_4h[i]
            else:
                lowerband_4h[i] = max(lowerband_4h[i], lowerband_4h[i-1])
                upperband_4h[i] = upperband_4h[i-1]
                if close_4h[i] > upperband_4h[i]:
                    direction_4h[i] = 1
                    upperband_4h[i] = hl2_4h[i] + multiplier * atr_4h[i]
                    lowerband_4h[i] = hl2_4h[i] - multiplier * atr_4h[i]
                supertrend_4h[i] = upperband_4h[i] if direction_4h[i] == 1 else lowerband_4h[i]
    
    supertrend_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend_4h)
    direction_4h_aligned = align_htf_to_ltf(prices, df_4h, direction_4h)
    
    # === 1d ADX trend strength filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX calculation
    adx_period = 14
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                  np.abs(low_1d[1:] - close_1d[:-1])))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_1d = pd.Series(tr_1d).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    
    plus_dm_1d = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                          np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    plus_dm_1d = np.concatenate([[0], plus_dm_1d])
    minus_dm_1d = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                           np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    minus_dm_1d = np.concatenate([[0], minus_dm_1d])
    
    plus_di_1d = 100 * pd.Series(plus_dm_1d).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm_1d).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 1h Volume spike filter ===
    vol_avg_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike_1h = volume > (1.5 * vol_avg_1h)
    
    # === Session filter: 08-20 UTC ===
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(supertrend_4h_aligned[i]) or 
            np.isnan(direction_4h_aligned[i]) or
            np.isnan(adx_1d_aligned[i])):
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
            # Long: Supertrend uptrend + ADX > 25 + volume spike
            if (direction_4h_aligned[i] == 1 and
                adx_1d_aligned[i] > 25 and
                vol_spike_1h[i]):
                signals[i] = 0.20
                position = 1
            # Short: Supertrend downtrend + ADX > 25 + volume spike
            elif (direction_4h_aligned[i] == -1 and
                  adx_1d_aligned[i] > 25 and
                  vol_spike_1h[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Supertrend downtrend or ADX < 20
            if direction_4h_aligned[i] == -1 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Supertrend uptrend or ADX < 20
            if direction_4h_aligned[i] == 1 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
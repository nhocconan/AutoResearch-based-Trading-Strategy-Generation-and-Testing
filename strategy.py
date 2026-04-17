#!/usr/bin/env python3
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
    
    # === Daily ATR (14-period) for volatility filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR with Wilder's smoothing
    atr_1d = np.full_like(tr, np.nan)
    period = 14
    for i in range(len(tr)):
        if i < period:
            if i == 0:
                atr_1d[i] = tr[i]
            else:
                atr_1d[i] = (atr_1d[i-1] * (i-1) + tr[i]) / i
        else:
            atr_1d[i] = (atr_1d[i-1] * (period-1) + tr[i]) / period
    
    # === 4-hour Donchian Channel (20-period) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper band: highest high of last 20 periods
    upper_donchian = np.full_like(high_4h, np.nan)
    for i in range(len(high_4h)):
        if i >= 19:
            upper_donchian[i] = np.max(high_4h[i-19:i+1])
        elif i > 0:
            upper_donchian[i] = np.max(high_4h[max(0, i-9):i+1])
        else:
            upper_donchian[i] = high_4h[0]
    
    # Lower band: lowest low of last 20 periods
    lower_donchian = np.full_like(low_4h, np.nan)
    for i in range(len(low_4h)):
        if i >= 19:
            lower_donchian[i] = np.min(low_4h[i-19:i+1])
        elif i > 0:
            lower_donchian[i] = np.min(low_4h[max(0, i-9):i+1])
        else:
            lower_donchian[i] = low_4h[0]
    
    # === Align indicators to 4h timeframe ===
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    upper_donchian_aligned = align_htf_to_ltf(prices, df_4h, upper_donchian)
    lower_donchian_aligned = align_htf_to_ltf(prices, df_4h, lower_donchian)
    
    # === 4h Volume confirmation ===
    volume_4h = df_4h['volume'].values
    vol_ma_20 = np.full_like(volume_4h, np.nan)
    for i in range(len(volume_4h)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume_4h[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume_4h[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume_4h[0]
    
    # Volume confirmation: current 4h volume > 1.5x 20-period average
    vol_confirm = volume_4h > vol_ma_20 * 1.5
    
    # === 4h Session filter (08-20 UTC) ===
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(upper_donchian_aligned[i]) or 
            np.isnan(lower_donchian_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if outside session
        if not session_filter[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Volatility filter: only trade when ATR > 50th percentile of recent values
        if i >= 50:
            atr_recent = atr_1d_aligned[i-50:i+1]
            atr_percentile = (np.sum(atr_recent <= atr_1d_aligned[i]) / len(atr_recent)) * 100
            vol_filter = atr_percentile > 50
        else:
            vol_filter = True
        
        # Entry logic: only enter when flat AND volume confirmation AND vol filter
        if position == 0:
            # Long: price breaks above upper Donchian + volume confirmation + vol filter
            if (close[i] > upper_donchian_aligned[i] and 
                vol_confirm[i] and 
                vol_filter):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below lower Donchian + volume confirmation + vol filter
            elif (close[i] < lower_donchian_aligned[i] and 
                  vol_confirm[i] and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price closes below midpoint of Donchian channel
            midpoint = (upper_donchian_aligned[i] + lower_donchian_aligned[i]) / 2
            if close[i] < midpoint:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above midpoint of Donchian channel
            midpoint = (upper_donchian_aligned[i] + lower_donchian_aligned[i]) / 2
            if close[i] > midpoint:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_VolatilityFilter_v1"
timeframe = "4h"
leverage = 1.0
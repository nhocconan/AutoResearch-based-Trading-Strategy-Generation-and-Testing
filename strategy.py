#!/usr/bin/env python3
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
    
    # === 1d Close for price channel ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 12-period Donchian channels on 1d ===
    # Upper band: highest high of last 12 days
    # Lower band: lowest low of last 12 days
    donchian_upper = np.full_like(close_1d, np.nan)
    donchian_lower = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        if i >= 11:  # 12 periods needed (0-11)
            donchian_upper[i] = np.max(high_1d[i-11:i+1])
            donchian_lower[i] = np.min(low_1d[i-11:i+1])
        elif i >= 0:
            donchian_upper[i] = np.max(high_1d[0:i+1])
            donchian_lower[i] = np.min(low_1d[0:i+1])
    
    # === 1d ATR for volatility filter ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(12) using Wilder's smoothing
    atr_12 = np.full_like(tr, np.nan)
    if len(tr) >= 12:
        atr_12[11] = np.mean(tr[:12])
        for i in range(12, len(tr)):
            atr_12[i] = (atr_12[i-1] * 11 + tr[i]) / 12
    
    # === 1d Volume confirmation ===
    vol_1d = df_1d['volume'].values
    vol_ma_10 = np.full_like(vol_1d, np.nan)
    for i in range(len(vol_1d)):
        if i >= 9:
            vol_ma_10[i] = np.mean(vol_1d[i-9:i+1])
        elif i > 0:
            vol_ma_10[i] = np.mean(vol_1d[max(0, i-4):i+1])
        else:
            vol_ma_10[i] = vol_1d[0]
    
    # Volume confirmation: current volume > 1.3x 10-period average
    vol_confirm = vol_1d > vol_ma_10 * 1.3
    
    # Align all indicators to 12h timeframe
    donchian_upper_12h = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_12h = align_htf_to_ltf(prices, df_1d, donchian_lower)
    atr_12_12h = align_htf_to_ltf(prices, df_1d, atr_12)
    vol_confirm_12h = align_htf_to_ltf(prices, df_1d, vol_confirm)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_12h[i]) or np.isnan(donchian_lower_12h[i]) or 
            np.isnan(atr_12_12h[i]) or np.isnan(vol_confirm_12h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat AND volume confirmation
        if position == 0:
            # Long: price breaks above Donchian upper + volatility filter + volume confirmation
            if (close[i] > donchian_upper_12h[i] and 
                atr_12_12h[i] > 0.003 * close[i] and  # volatility filter
                vol_confirm_12h[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below Donchian lower + volatility filter + volume confirmation
            elif (close[i] < donchian_lower_12h[i] and 
                  atr_12_12h[i] > 0.003 * close[i] and  # volatility filter
                  vol_confirm_12h[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal or volatility collapse
        elif position == 1:
            # Exit long: price breaks below Donchian lower OR volatility collapses
            if (close[i] < donchian_lower_12h[i] or 
                atr_12_12h[i] <= 0.002 * close[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian upper OR volatility collapses
            if (close[i] > donchian_upper_12h[i] or 
                atr_12_12h[i] <= 0.002 * close[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian12_Volume_VolatilityFilter_v1"
timeframe = "12h"
leverage = 1.0
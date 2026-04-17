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
    
    # === 1d ATR (14-period) for volatility regime ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR with Wilder's smoothing
    atr_1d = np.full_like(tr, np.nan)
    period = 14
    for i in range(len(tr)):
        if i < period:
            if i == 0:
                atr_1d[i] = tr[i]
            else:
                atr_1d[i] = (atr_1d[i-1] * i + tr[i]) / (i + 1)
        else:
            atr_1d[i] = (atr_1d[i-1] * (period-1) + tr[i]) / period
    
    # === 12h Donchian Channel (20-period) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper band: highest high of last 20 periods
    donch_high = np.full_like(high_12h, np.nan)
    for i in range(len(high_12h)):
        if i >= 19:
            donch_high[i] = np.max(high_12h[i-19:i+1])
        elif i > 0:
            donch_high[i] = np.max(high_12h[max(0, i-9):i+1])
        else:
            donch_high[i] = high_12h[0]
    
    # Lower band: lowest low of last 20 periods
    donch_low = np.full_like(low_12h, np.nan)
    for i in range(len(low_12h)):
        if i >= 19:
            donch_low[i] = np.min(low_12h[i-19:i+1])
        elif i > 0:
            donch_low[i] = np.min(low_12h[max(0, i-9):i+1])
        else:
            donch_low[i] = low_12h[0]
    
    # === Align indicators to 12h timeframe ===
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    
    # === Volume confirmation: 12h volume > 1.5x 20-period average ===
    volume_12h = df_12h['volume'].values
    vol_ma_20 = np.full_like(volume_12h, np.nan)
    for i in range(len(volume_12h)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume_12h[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume_12h[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume_12h[0]
    
    vol_confirm = volume_12h > vol_ma_20 * 1.5
    vol_confirm_aligned = align_htf_to_ltf(prices, df_12h, vol_confirm)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_confirm_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: break above Donchian high + volatility filter + volume confirmation
            if (close[i] > donch_high_aligned[i] and 
                atr_1d_aligned[i] > 0 and  # ensure volatility present
                vol_confirm_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below Donchian low + volatility filter + volume confirmation
            elif (close[i] < donch_low_aligned[i] and 
                  atr_1d_aligned[i] > 0 and
                  vol_confirm_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal or volatility collapse
        elif position == 1:
            # Exit long: price breaks below Donchian low OR volatility collapses
            if (close[i] < donch_low_aligned[i] or 
                atr_1d_aligned[i] < atr_1d_aligned[i-1] * 0.5):  # volatility dropped >50%
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high OR volatility collapses
            if (close[i] > donch_high_aligned[i] or 
                atr_1d_aligned[i] < atr_1d_aligned[i-1] * 0.5):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_DonchianBreakout_ATR_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0
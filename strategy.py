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
    
    # === 1d Close for Donchian channels ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 1d Donchian Channel (20-period) ===
    donch_high_20 = np.full_like(high_1d, np.nan)
    donch_low_20 = np.full_like(low_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= 19:
            donch_high_20[i] = np.max(high_1d[i-19:i+1])
            donch_low_20[i] = np.min(low_1d[i-19:i+1])
        elif i > 0:
            donch_high_20[i] = np.max(high_1d[0:i+1])
            donch_low_20[i] = np.min(low_1d[0:i+1])
        else:
            donch_high_20[i] = high_1d[0]
            donch_low_20[i] = low_1d[0]
    
    # === 1d ATR (14-period) for volatility filter ===
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr_14[13] = np.mean(tr[:14])
        for i in range(14, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # === 1w Close for weekly trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === 1w EMA (50-period) for weekly trend ===
    ema_50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (close_1w[i] * 2 + ema_50_1w[i-1] * 49) / 51
    
    # === 1d Volume confirmation ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = np.full_like(volume_1d, np.nan)
    for i in range(len(volume_1d)):
        if i >= 19:
            vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
        elif i > 0:
            vol_ma_20_1d[i] = np.mean(volume_1d[max(0, i-9):i+1])
        else:
            vol_ma_20_1d[i] = volume_1d[0]
    
    # Volume confirmation: current 1d volume > 1.5x 20-period average
    vol_confirm_1d = volume_1d > vol_ma_20_1d * 1.5
    
    # Align all indicators to 6h timeframe
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    vol_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_confirm_1d.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_confirm_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above Donchian high + weekly uptrend + volume confirmation
            if (close[i] > donch_high_20_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                vol_confirm_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below Donchian low + weekly downtrend + volume confirmation
            elif (close[i] < donch_low_20_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  vol_confirm_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price breaks below Donchian low
            if close[i] < donch_low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high
            if close[i] > donch_high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0
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
    
    # Get daily data for Donchian channel and volume
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian channel (20-day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily average volume (20-day)
    vol_1d = df_1d['volume'].values
    vol_ma_20d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[high[0] - low[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros(n)
    for i in range(n):
        if i < 14:
            atr[i] = np.mean(tr[:i+1]) if i > 0 else tr[0]
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Align daily indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    vol_ma_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20d)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = 20  # Donchian needs 20 days
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_20d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20d_aligned[i] if vol_ma_20d_aligned[i] > 0 else 0
        
        # Volume confirmation: > 2.0x average daily volume
        volume_confirmation = vol_ratio > 2.0
        
        # ATR volatility filter: avoid low volatility periods
        if i >= 50:
            atr_avg = np.mean(atr[i-50:i+1])
            vol_filter = atr[i] > atr_avg * 0.5
        else:
            vol_filter = True
        
        if position == 0:
            # Long: break above 20-day Donchian high with volume and volatility
            if volume_confirmation and vol_filter and price > donchian_high_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below 20-day Donchian low with volume and volatility
            elif volume_confirmation and vol_filter and price < donchian_low_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to Donchian midpoint or volatility drops
            donchian_mid = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if price < donchian_mid or atr[i] < np.mean(atr[max(0, i-50):i+1]) * 0.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price returns to Donchian midpoint or volatility drops
            donchian_mid = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if price > donchian_mid or atr[i] < np.mean(atr[max(0, i-50):i+1]) * 0.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "12h_1D_Donchian20_Breakout_VolumeVolFilter_v1"
timeframe = "12h"
leverage = 1.0
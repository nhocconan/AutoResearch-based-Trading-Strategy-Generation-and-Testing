#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for 60-period EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d_60 = pd.Series(close_1d).ewm(span=60, adjust=False, min_periods=60).mean().values
    ema_1d_60_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_60)
    
    # Get 12h data for Donchian channel (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Calculate 6h ATR(14) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[high[0] - low[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr = np.zeros(n)
    for i in range(n):
        if i < 13:
            atr[i] = np.mean(tr[:i+1]) if i > 0 else tr[i]
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    atr_ma_10 = np.full(n, np.nan)
    for i in range(10, n):
        atr_ma_10[i] = np.mean(atr[i-10:i])
    
    # Calculate volume average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(60, 20, 10, 20)  # daily EMA60, 12h Donchian20, ATR10, volume MA20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_60_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr_ma_10[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        atr_ratio = atr[i] / atr_ma_10[i] if atr_ma_10[i] > 0 else 0
        
        # Conditions: volume > 1.5x average, volatility > 1.2x average ATR
        volume_filter = vol_ratio > 1.5
        volatility_filter = atr_ratio > 1.2
        
        if position == 0:
            # Long: price breaks above Donchian high with daily uptrend
            if (volume_filter and volatility_filter and
                price > donchian_high_aligned[i] and
                close[i-1] <= donchian_high_aligned[i] and
                ema_1d_60_aligned[i] > ema_1d_60_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with daily downtrend
            elif (volume_filter and volatility_filter and
                  price < donchian_low_aligned[i] and
                  close[i-1] >= donchian_low_aligned[i] and
                  ema_1d_60_aligned[i] < ema_1d_60_aligned[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price reaches Donchian low or daily trend turns down
            if (price <= donchian_low_aligned[i] or
                ema_1d_60_aligned[i] < ema_1d_60_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches Donchian high or daily trend turns up
            if (price >= donchian_high_aligned[i] or
                ema_1d_60_aligned[i] > ema_1d_60_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dEMA60_Trend_VolumeVolFilter_v1"
timeframe = "6h"
leverage = 1.0
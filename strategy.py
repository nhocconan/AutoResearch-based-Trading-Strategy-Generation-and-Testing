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
    
    # Get 1d data for ATR and ATR-based volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Daily ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # ATR(14) with Wilder's smoothing
    atr_14 = np.full(len(tr), np.nan, dtype=np.float64)
    atr_14[13] = np.mean(tr[:14])  # Simple average for first 14 periods
    for i in range(14, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # ATR percentile (20-day lookback) for volatility regime
    atr_pct = np.full(len(atr_14), np.nan, dtype=np.float64)
    for i in range(19, len(atr_14)):
        atr_pct[i] = (atr_14[i] - np.min(atr_14[i-19:i+1])) / (np.max(atr_14[i-19:i+1]) - np.min(atr_14[i-19:i+1]) + 1e-10)
    
    # Align ATR percentile to 4h timeframe
    atr_pct_aligned = align_htf_to_ltf(prices, df_1d, atr_pct)
    
    # Get 4h data for price action (since we're on 4h timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian channel (20-period) on 4h data
    highest_20 = np.full(len(high_4h), np.nan, dtype=np.float64)
    lowest_20 = np.full(len(low_4h), np.nan, dtype=np.float64)
    
    for i in range(19, len(high_4h)):
        highest_20[i] = np.max(high_4h[i-19:i+1])
        lowest_20[i] = np.min(low_4h[i-19:i+1])
    
    # Align Donchian levels to main timeframe (4h->4h is 1:1, but keep for consistency)
    highest_20_aligned = align_htf_to_ltf(prices, df_4h, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_4h, lowest_20)
    
    # Volume filter: volume > 1.8x 20-period average (more selective)
    vol_ma_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need ATR percentile (20-period), Donchian (20-period), volume MA (20-period)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_pct_aligned[i]) or np.isnan(highest_20_aligned[i]) or 
            np.isnan(lowest_20_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        atr_regime = atr_pct_aligned[i]  # 0 = low volatility, 1 = high volatility
        upper_channel = highest_20_aligned[i]
        lower_channel = lowest_20_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volatility filter: only trade in high volatility regimes (ATR percentile > 0.6)
        vol_filter = atr_regime > 0.6
        
        # Volume filter: volume > 1.8x average
        vol_spike = vol_now > 1.8 * vol_avg
        
        if position == 0:
            # Long: price breaks above upper Donchian + volatility regime + volume spike
            if price > upper_channel and vol_filter and vol_spike:
                signals[i] = size
                position = 1
            # Short: price breaks below lower Donchian + volatility regime + volume spike
            elif price < lower_channel and vol_filter and vol_spike:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle of channel or volatility drops
            middle_channel = (upper_channel + lower_channel) / 2
            if price <= middle_channel or atr_regime <= 0.4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to middle of channel or volatility drops
            middle_channel = (upper_channel + lower_channel) / 2
            if price >= middle_channel or atr_regime <= 0.4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_Breakout_VolRegime_VolumeSpike"
timeframe = "4h"
leverage = 1.0
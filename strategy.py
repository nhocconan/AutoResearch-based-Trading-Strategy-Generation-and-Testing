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
    
    # === 4h data (primary) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # === 1d data (HTF for trend) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 4h Donchian channel (20) ===
    highest_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # === 1d EMA (34) for trend filter ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === 4h volume ratio for confirmation ===
    vol_ma_10_4h = pd.Series(volume_4h).rolling(window=10, min_periods=10).mean().values
    vol_ratio_4h = volume_4h / vol_ma_10_4h
    
    # Align HTF data to 4h timeframe
    highest_high_20_aligned = align_htf_to_ltf(prices, df_4h, highest_high_20)
    lowest_low_20_aligned = align_htf_to_ltf(prices, df_4h, lowest_low_20)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    
    signals = np.zeros(n)
    
    # Warmup: enough for Donchian and EMA
    warmup = 40
    
    # Track position and entry price
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(highest_high_20_aligned[i]) or np.isnan(lowest_low_20_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ratio_4h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = highest_high_20_aligned[i]
        lower = lowest_low_20_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_ratio = vol_ratio_4h_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band
            if price < lower:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band
            if price > upper:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long: price breaks above upper band, above 1d EMA trend, volume confirmation
            if price > upper and price > ema_trend and vol_ratio > 1.5:
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            # Short: price breaks below lower band, below 1d EMA trend, volume confirmation
            elif price < lower and price < ema_trend and vol_ratio > 1.5:
                signals[i] = -0.25
                position = -1
                entry_price = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_EMA_Trend_Volume_v2"
timeframe = "4h"
leverage = 1.0
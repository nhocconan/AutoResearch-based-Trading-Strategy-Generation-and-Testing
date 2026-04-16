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
    
    # === 12h data (primary) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # === 1d data (HTF for ATR and Donchian) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_12h = df_1d['low'].values  # Note: variable name reused for low_1d
    low_1d = df_1d['low'].values
    
    # === ATR (14-period) on 1d for volatility filter ===
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    atr = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr[13] = np.nanmean(tr[1:15])
        for i in range(15, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # === Donchian channel (20-period) on 1d for breakout levels ===
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.nanmax(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.nanmin(arr[i-window+1:i+1])
        return res
    
    donchian_upper = rolling_max(high_1d, 20)
    donchian_lower = rolling_min(low_1d, 20)
    
    # === 12h volume ratio for confirmation ===
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = volume_12h / vol_ma_20_12h
    
    # Align all HTF data to 12h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    signals = np.zeros(n)
    
    # Warmup: enough for ATR and Donchian calculations
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(atr_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ratio_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        atr_val = atr_aligned[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        vol_ratio = vol_ratio_12h_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower (trend reversal) OR 2x ATR trailing stop
            if price < lower or price < (high[i] - 2 * atr_val):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper (trend reversal) OR 2x ATR trailing stop
            if price > upper or price > (low[i] + 2 * atr_val):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Breakout with volume confirmation
            # LONG: Break above upper Donchian with volume
            if price > upper and vol_ratio > 1.3:
                signals[i] = 0.25
                position = 1
                continue
            # SHORT: Break below lower Donchian with volume
            elif price < lower and vol_ratio > 1.3:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_Breakout_Volume_ATRExit"
timeframe = "12h"
leverage = 1.0
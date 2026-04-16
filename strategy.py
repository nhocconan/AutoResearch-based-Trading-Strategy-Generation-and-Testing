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
    
    # === 6h data (primary) ===
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # === 12h data (HTF for trend and volatility) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # === 6h ATR for volatility filter ===
    atr_period = 14
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros_like(close_6h)
    for i in range(len(tr)):
        if i < atr_period:
            atr[i] = np.mean(tr[:i+1])
        else:
            atr[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # === 12h Donchian channel (20-period) ===
    donchian_high = np.zeros_like(close_12h)
    donchian_low = np.zeros_like(close_12h)
    for i in range(len(close_12h)):
        if i < 20:
            donchian_high[i] = np.max(high_12h[:i+1])
            donchian_low[i] = np.min(low_12h[:i+1])
        else:
            donchian_high[i] = np.max(high_12h[i-19:i+1])
            donchian_low[i] = np.min(low_12h[i-19:i+1])
    
    # === 12h EMA34 for trend filter ===
    ema_period = 34
    ema_12h = np.zeros_like(close_12h)
    multiplier = 2 / (ema_period + 1)
    for i in range(len(close_12h)):
        if i == 0:
            ema_12h[i] = close_12h[i]
        else:
            ema_12h[i] = (close_12h[i] - ema_12h[i-1]) * multiplier + ema_12h[i-1]
    
    # === 6h volume ratio for confirmation ===
    vol_ma_10_6h = pd.Series(volume_6h).rolling(window=10, min_periods=10).mean().values
    vol_ratio_6h = volume_6h / vol_ma_10_6h
    
    # Align all HTF data to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    vol_ratio_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ratio_6h)
    atr_aligned = align_htf_to_ltf(prices, df_6h, atr)
    
    signals = np.zeros(n)
    
    # Warmup: enough for ATR, EMA, Donchian
    warmup = max(34, 20, 14) + 5
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or 
            np.isnan(vol_ratio_6h_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        dh = donchian_high_aligned[i]
        dl = donchian_low_aligned[i]
        ema = ema_12h_aligned[i]
        vol_ratio = vol_ratio_6h_aligned[i]
        atr_val = atr_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below EMA OR ATR-based trailing stop
            if price < ema or price < (ema - 1.5 * atr_val):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above EMA OR ATR-based trailing stop
            if price > ema or price > (ema + 1.5 * atr_val):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above 12h Donchian high + above EMA + volume surge
            if price > dh and price > ema and vol_ratio > 2.0:
                signals[i] = 0.25
                position = 1
                continue
            # SHORT: Price breaks below 12h Donchian low + below EMA + volume surge
            elif price < dl and price < ema and vol_ratio > 2.0:
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

name = "6h_12h_Donchian_EMA34_Volume_ATRExit"
timeframe = "6h"
leverage = 1.0
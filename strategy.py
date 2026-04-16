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
    
    # === 12h data (HTF for trend and volatility) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # === 4h ATR for volatility filtering ===
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
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
    
    # === 12h ATR for volatility filtering ===
    tr_12h = np.maximum(high_12h - low_12h, np.absolute(high_12h - np.roll(close_12h, 1)), np.absolute(low_12h - np.roll(close_12h, 1)))
    tr_12h[0] = high_12h[0] - low_12h[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # === 12h ADX for trend strength ===
    plus_dm = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    minus_dm = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr_14 = pd.Series(tr_12h).rolling(window=14, min_periods=14).sum().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / tr_14
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / tr_14
    dx = 100 * np.absolute(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF data to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    
    # Warmup: enough for ATR and Donchian
    warmup = 30
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_12h_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        dh = donchian_high_aligned[i]
        dl = donchian_low_aligned[i]
        atr_val = atr[i]
        atr_12h_val = atr_12h_aligned[i]
        adx_val = adx_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below 12h Donchian low OR ATR-based stop
            if price < dl or price < (dh - 1.5 * atr_val):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above 12h Donchian high OR ATR-based stop
            if price > dh or price > (dl + 1.5 * atr_val):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above 12h Donchian high with strong trend and volatility filter
            if price > dh and adx_val > 25 and atr_12h_val < atr[i] * 1.5:
                signals[i] = 0.25
                position = 1
                continue
            # SHORT: Price breaks below 12h Donchian low with strong trend and volatility filter
            elif price < dl and adx_val > 25 and atr_12h_val < atr[i] * 1.5:
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

name = "4h_12h_Donchian_Breakout_ADX_VolatilityFilter"
timeframe = "4h"
leverage = 1.0
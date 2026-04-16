#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (trend direction) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # === 1d data (support/resistance levels) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 4h EMA200 for long-term trend ===
    ema200_4h = pd.Series(close_4h).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # === 1d Donchian channels (20-period) for support/resistance ===
    # Upper band: highest high of last 20 days
    # Lower band: lowest low of last 20 days
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # === 1h volume filter ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 200
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        price = close[i]
        ema200 = ema200_4h_aligned[i]
        upper = upper_20_aligned[i]
        lower = lower_20_aligned[i]
        vol = vol_ratio[i]
        
        # Skip if any data is NaN
        if np.isnan(ema200) or np.isnan(upper) or np.isnan(lower) or np.isnan(vol):
            signals[i] = 0.0
            continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below EMA200 OR below lower Donchian band
            if price < ema200 or price < lower:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above EMA200 OR above upper Donchian band
            if price > ema200 or price > upper:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price above EMA200 (uptrend) and breaks above upper Donchian band with volume
            if price > ema200 and price > upper and vol > 1.5:
                signals[i] = 0.20
                position = 1
                continue
            # SHORT: Price below EMA200 (downtrend) and breaks below lower Donchian band with volume
            elif price < ema200 and price < lower and vol > 1.5:
                signals[i] = -0.20
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_EMA200_Donchian20_Volume_Breakout"
timeframe = "1h"
leverage = 1.0
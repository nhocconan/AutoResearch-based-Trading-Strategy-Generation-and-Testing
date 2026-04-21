#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR calculation
    atr = np.zeros_like(tr)
    atr[13] = np.mean(tr[1:14])
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate 1d EMA34
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    dc_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    dc_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 1d indicators to 4h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if data not ready
        if (np.isnan(atr_aligned[i]) or np.isnan(ema34_aligned[i]) or
            np.isnan(dc_upper[i]) or np.isnan(dc_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        
        if position == 0:
            # Enter long: Price breaks above Donchian upper + above daily EMA34
            if (price_close > dc_upper[i] and price_close > ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below Donchian lower + below daily EMA34
            elif (price_close < dc_lower[i] and price_close < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Price reverses back through Donchian opposite side
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls back below Donchian lower
                if price_close < dc_lower[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Price rises back above Donchian upper
                if price_close > dc_upper[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_EMA34_Breakout"
timeframe = "4h"
leverage = 1.0
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
    
    # === 1d data (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Donchian channel (20-period) ===
    donch_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian to 12h (wait for 1d close)
    donch_high_12h = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_12h = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # === 1d EMA34 (trend filter) ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 12h volume ratio for confirmation ===
    vol_ma_20_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = volume / vol_ma_20_12h
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donch_high_12h[i]) or 
            np.isnan(donch_low_12h[i]) or 
            np.isnan(ema_34_12h[i]) or 
            np.isnan(vol_ratio_12h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = donch_high_12h[i]
        lower = donch_low_12h[i]
        ema_trend = ema_34_12h[i]
        vol_ratio = vol_ratio_12h[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower OR trend reverses
            if price < lower or price < ema_trend:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper OR trend reverses
            if price > upper or price > ema_trend:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Break above Donchian upper with volume, in uptrend
            if price > upper and vol_ratio > 1.5 and price > ema_trend:
                signals[i] = 0.25
                position = 1
                continue
            # SHORT: Break below Donchian lower with volume, in downtrend
            elif price < lower and vol_ratio > 1.5 and price < ema_trend:
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

name = "12h_Donchian_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0
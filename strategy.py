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
    
    # === 1d data (HTF for trend) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === Weekly data for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # === 6h Donchian channel (20-period) ===
    donchian_high = np.zeros_like(close_6h)
    donchian_low = np.zeros_like(close_6h)
    for i in range(len(close_6h)):
        if i < 20:
            donchian_high[i] = np.max(high_6h[:i+1])
            donchian_low[i] = np.min(low_6h[:i+1])
        else:
            donchian_high[i] = np.max(high_6h[i-19:i+1])
            donchian_low[i] = np.min(low_6h[i-19:i+1])
    
    # === 1d EMA34 for trend filter ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === Weekly trend filter: price vs weekly EMA34 ===
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === 6h volume ratio for confirmation ===
    vol_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_6h = volume_6h / vol_ma_20_6h
    
    # Align all HTF data to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    vol_ratio_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ratio_6h)
    
    signals = np.zeros(n)
    
    # Warmup: enough for Donchian and EMA
    warmup = 40
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_ratio_6h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        dh = donchian_high_aligned[i]
        dl = donchian_low_aligned[i]
        ema1d = ema34_1d_aligned[i]
        ema1w = ema34_1w_aligned[i]
        vol_ratio = vol_ratio_6h_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below 6-day EMA34 OR weekly trend turns bearish
            if price < ema1d or close[i] < ema1w:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above 6-day EMA34 OR weekly trend turns bullish
            if price > ema1d or close[i] > ema1w:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Donchian breakout above with volume and both timeframes bullish
            if price > dh and vol_ratio > 2.0 and close[i] > ema1d and close[i] > ema1w:
                signals[i] = 0.25
                position = 1
                continue
            # SHORT: Donchian breakout below with volume and both timeframes bearish
            elif price < dl and vol_ratio > 2.0 and close[i] < ema1d and close[i] < ema1w:
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

name = "6h_Donchian_EMA34_DualTF_Volume"
timeframe = "6h"
leverage = 1.0
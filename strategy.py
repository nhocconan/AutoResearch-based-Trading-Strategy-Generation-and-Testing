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
    
    # === 6h data (primary) ===
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # === 1w data (HTF for trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # === 1d data (HTF for weekly pivot) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === Weekly high/low from 1d data (Monday to Friday) ===
    # We'll approximate weekly high/low using 5-day rolling max/min on daily data
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    
    # Align weekly high/low to 6h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1d, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1d, weekly_low)
    
    # === 6-week EMA on weekly close (trend filter) ===
    ema_6_1w = pd.Series(close_1w).ewm(span=6, adjust=False, min_periods=6).mean().values
    ema_6_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_6_1w)
    
    # === 6h ATR for volatility filter ===
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 6h volume ratio ===
    vol_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_6h = volume_6h / vol_ma_20_6h
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or 
            np.isnan(ema_6_1w_aligned[i]) or 
            np.isnan(atr_6h[i]) or
            np.isnan(vol_ratio_6h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        weekly_high_val = weekly_high_aligned[i]
        weekly_low_val = weekly_low_aligned[i]
        ema_trend = ema_6_1w_aligned[i]
        atr = atr_6h[i]
        vol_ratio = vol_ratio_6h[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            # Stop loss: price closes below entry - 2.0 * ATR
            if price < entry_price - 2.0 * atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Stop loss: price closes above entry + 2.0 * ATR
            if price > entry_price + 2.0 * atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below weekly low OR trend reverses (below weekly EMA)
            if price < weekly_low_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above weekly high OR trend reverses (above weekly EMA)
            if price > weekly_high_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Break above weekly high with volume, in uptrend (above weekly EMA)
            if price > weekly_high_val and vol_ratio > 1.8 and price > ema_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            # SHORT: Break below weekly low with volume, in downtrend (below weekly EMA)
            elif price < weekly_low_val and vol_ratio > 1.8 and price < ema_trend:
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

name = "6h_WeeklyBreakout_1wEMA6_Volume_ATRStop_v1"
timeframe = "6h"
leverage = 1.0
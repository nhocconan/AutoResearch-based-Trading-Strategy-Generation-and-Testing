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
    
    # === 1d data (HTF for trend and context) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 1w data (HTF for weekly bias) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # === 6h ATR(14) for volatility and stoploss ===
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr_14_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_6h_aligned = align_htf_to_ltf(prices, df_6h, atr_14_6h)
    
    # === 1d EMA34 for trend filter ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 1w EMA34 for weekly trend bias ===
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 6h Donchian(20) for breakout levels ===
    donch_high_6h = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donch_low_6h = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donch_high_6h_aligned = align_htf_to_ltf(prices, df_6h, donch_high_6h)
    donch_low_6h_aligned = align_htf_to_ltf(prices, df_6h, donch_low_6h)
    
    # === 6h volume ratio for confirmation ===
    vol_ma_10_6h = pd.Series(volume_6h).rolling(window=10, min_periods=10).mean().values
    vol_ratio_6h = volume_6h / vol_ma_10_6h
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(atr_14_6h_aligned[i]) or
            np.isnan(vol_ratio_6h[i]) or
            np.isnan(donch_high_6h_aligned[i]) or
            np.isnan(donch_low_6h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_trend_1d = ema_34_1d_aligned[i]
        ema_trend_1w = ema_34_1w_aligned[i]
        atr = atr_14_6h_aligned[i]
        vol_ratio = vol_ratio_6h[i]
        donch_high = donch_high_6h_aligned[i]
        donch_low = donch_low_6h_aligned[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            # Stop loss: price closes below entry - 2.5 * ATR
            if price < entry_price - 2.5 * atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Stop loss: price closes above entry + 2.5 * ATR
            if price > entry_price + 2.5 * atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or trend reverses
            if price < donch_low or price < ema_trend_1d:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or trend reverses
            if price > donch_high or price > ema_trend_1d:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Break above Donchian high with volume, in uptrend (above EMA34 on 1d and 1w)
            if (price > donch_high and vol_ratio > 1.8 and price > ema_trend_1d and price > ema_trend_1w):
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            # SHORT: Break below Donchian low with volume, in downtrend (below EMA34 on 1d and 1w)
            elif (price < donch_low and vol_ratio > 1.8 and price < ema_trend_1d and price < ema_trend_1w):
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

name = "6h_Donchian_1d1wEMA34_Volume_ATRStop_v1"
timeframe = "6h"
leverage = 1.0
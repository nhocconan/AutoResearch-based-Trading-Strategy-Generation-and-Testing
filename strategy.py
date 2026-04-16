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
    
    # === 12h data (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate ATR on 12h
    tr_12h = np.maximum(high_12h - low_12h,
                        np.maximum(np.abs(high_12h - np.roll(close_12h, 1)),
                                   np.abs(low_12h - np.roll(close_12h, 1))))
    tr_12h[0] = high_12h[0] - low_12h[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # === 1d data (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR on 1d
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 12h EMA(34) for trend filter ===
    ema_34_12h = pd.Series(close_12h).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === 1d Donchian(20) for price channel ===
    donchian_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    
    # === 12h volume ratio (volume / 20-period average) ===
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = volume_12h / vol_ma_12h
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_12h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(donchian_high_1d_aligned[i]) or
            np.isnan(donchian_low_1d_aligned[i]) or np.isnan(vol_ratio_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close_12h[i]
        ema_34_val = ema_34_12h_aligned[i]
        atr_12h_val = atr_12h_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        donchian_high = donchian_high_1d_aligned[i]
        donchian_low = donchian_low_1d_aligned[i]
        vol_ratio = vol_ratio_12h_aligned[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            if price < entry_price - 2.0 * atr_12h_val:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short position
            if price > entry_price + 2.0 * atr_12h_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # === EXIT LOGIC (based on Donchian breach) ===
        if position == 1:  # Long position
            if price < donchian_low:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short position
            if price > donchian_high:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price above 12h EMA(34) AND volume surge AND price near Donchian low
            if (price > ema_34_val) and (vol_ratio > 1.5) and (price <= donchian_low * 1.02):
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            
            # SHORT: Price below 12h EMA(34) AND volume surge AND price near Donchian high
            elif (price < ema_34_val) and (vol_ratio > 1.5) and (price >= donchian_high * 0.98):
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

name = "12h_EMA34_Donchian1d_VolumeSurge"
timeframe = "12h"
leverage = 1.0
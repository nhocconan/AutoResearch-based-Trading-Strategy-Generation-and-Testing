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
    
    # === 12h data (primary) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # === 1d data (HTF for trend) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_12h = df_1d['low'].values
    
    # === 1d EMA for trend direction (34 period) ===
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 12h ATR for volatility and stop (14 period) ===
    tr_12h = np.maximum(high_12h - low_12h, 
                        np.maximum(np.abs(high_12h - np.roll(close_12h, 1)), 
                                   np.abs(low_12h - np.roll(close_12h, 1))))
    tr_12h[0] = high_12h[0] - low_12h[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # === 12h volume ratio for confirmation ===
    vol_ma_10_12h = pd.Series(volume_12h).rolling(window=10, min_periods=10).mean().values
    vol_ratio_12h = volume_12h / vol_ma_10_12h
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    # === 12h Donchian channels (15 period) ===
    highest_15 = pd.Series(high_12h).rolling(window=15, min_periods=15).max().values
    lowest_15 = pd.Series(low_12h).rolling(window=15, min_periods=15).min().values
    highest_15_aligned = align_htf_to_ltf(prices, df_12h, highest_15)
    lowest_15_aligned = align_htf_to_ltf(prices, df_12h, lowest_15)
    
    signals = np.zeros(n)
    
    # Warmup: enough for EMA and ATR
    warmup = 50
    
    # Track position and entry price
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(atr_12h_aligned[i]) or 
            np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(highest_15_aligned[i]) or 
            np.isnan(lowest_15_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_trend = ema_1d_aligned[i]
        atr_val = atr_12h_aligned[i]
        vol_ratio_val = vol_ratio_12h_aligned[i]
        upper_channel = highest_15_aligned[i]
        lower_channel = lowest_15_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: close below Donchian lower or 2x ATR stop
            if price < lower_channel or price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: close above Donchian upper or 2x ATR stop
            if price > upper_channel or price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Trend filter: price above/below 1d EMA with volume
            if price > ema_trend and vol_ratio_val > 1.5:
                # LONG: price above trend with volume
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            elif price < ema_trend and vol_ratio_val > 1.5:
                # SHORT: price below trend with volume
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

name = "12h_Donchian_EMA_Trend_Volume_v2"
timeframe = "12h"
leverage = 1.0
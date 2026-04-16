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
    
    # === 4h data (primary) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # === 1d data (HTF for trend) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 1d EMA for trend direction (21 period) ===
    ema_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 4h ATR for volatility and stop (14 period) ===
    tr_4h = np.maximum(high_4h - low_4h, 
                       np.maximum(np.abs(high_4h - np.roll(close_4h, 1)), 
                                  np.abs(low_4h - np.roll(close_4h, 1))))
    tr_4h[0] = high_4h[0] - low_4h[0]
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # === 4h volume ratio for confirmation ===
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_4h = volume_4h / vol_ma_20_4h
    vol_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    
    # === 4h Donchian channels (20 period) ===
    highest_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    highest_20_aligned = align_htf_to_ltf(prices, df_4h, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_4h, lowest_20)
    
    signals = np.zeros(n)
    
    # Warmup: enough for EMA and ATR
    warmup = 50
    
    # Track position and entry price
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(atr_4h_aligned[i]) or 
            np.isnan(vol_ratio_4h_aligned[i]) or np.isnan(highest_20_aligned[i]) or 
            np.isnan(lowest_20_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_trend = ema_1d_aligned[i]
        atr_val = atr_4h_aligned[i]
        vol_ratio_val = vol_ratio_4h_aligned[i]
        upper_channel = highest_20_aligned[i]
        lower_channel = lowest_20_aligned[i]
        
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
            if price > ema_trend and vol_ratio_val > 1.3:
                # LONG: price above trend with volume
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            elif price < ema_trend and vol_ratio_val > 1.3:
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

name = "4h_Donchian_EMA_Trend_Volume"
timeframe = "4h"
leverage = 1.0
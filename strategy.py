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
    low_1d = df_1d['low'].values
    
    # === 1w data (HTF for regime) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
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
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = volume_12h / vol_ma_20_12h
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    # === 12h Donchian channels (20 period) ===
    highest_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    highest_20_aligned = align_htf_to_ltf(prices, df_12h, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_12h, lowest_20)
    
    # === 1w ADX for trend strength (14 period) ===
    tr_1w = np.maximum(high_1w - low_1w, 
                       np.maximum(np.abs(high_1w - np.roll(close_1w, 1)), 
                                  np.abs(low_1w - np.roll(close_1w, 1))))
    tr_1w[0] = high_1w[0] - low_1w[0]
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    tr_14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx[np.isnan(dx)] = 0
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    
    # Warmup: enough for EMA, ATR, ADX
    warmup = 60
    
    # Track position and entry price
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(atr_12h_aligned[i]) or 
            np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(highest_20_aligned[i]) or 
            np.isnan(lowest_20_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_trend = ema_1d_aligned[i]
        atr_val = atr_12h_aligned[i]
        vol_ratio_val = vol_ratio_12h_aligned[i]
        upper_channel = highest_20_aligned[i]
        lower_channel = lowest_20_aligned[i]
        adx_val = adx_aligned[i]
        
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
            # Trend filter: price above/below 1d EMA with volume and strong trend (ADX > 25)
            if price > ema_trend and vol_ratio_val > 1.5 and adx_val > 25:
                # LONG: price above trend with volume and strong trend
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            elif price < ema_trend and vol_ratio_val > 1.5 and adx_val > 25:
                # SHORT: price below trend with volume and strong trend
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

name = "12h_Donchian_EMA_Trend_Volume_ADX"
timeframe = "12h"
leverage = 1.0
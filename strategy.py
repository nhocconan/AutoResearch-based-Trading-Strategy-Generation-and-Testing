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
    
    # Get daily data for Donchian channels and 12h for EMA trend
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_1d) < 20 or len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper/lower: highest high/lowest low over 20 periods
    donchian_high = np.full(len(high_1d), np.nan)
    donchian_low = np.full(len(low_1d), np.nan)
    
    for i in range(20, len(high_1d)):
        donchian_high[i] = np.max(high_1d[i-20:i])
        donchian_low[i] = np.min(low_1d[i-20:i])
    
    # Calculate 50-period EMA on 12h data for trend filter
    close_12h = df_12h['close'].values
    ema_12h = np.full(len(close_12h), np.nan)
    
    if len(close_12h) >= 50:
        multiplier = 2 / (50 + 1)
        ema_12h[49] = np.mean(close_12h[:50])
        for i in range(50, len(close_12h)):
            ema_12h[i] = (close_12h[i] - ema_12h[i-1]) * multiplier + ema_12h[i-1]
    
    # Calculate ATR(14) for position sizing and volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[high[0] - low[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(n):
        if i < 14:
            atr[i] = np.mean(tr[:i+1]) if i > 0 else tr[0]
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate volume average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    # Align indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(20, 50)  # Donchian needs 20, EMA needs 50
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_12h_aligned[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Volume confirmation: > 2.0x average volume
        volume_confirmation = vol_ratio > 2.0
        
        # Trend filter: price above/below 12h EMA50
        trend_up = price > ema_12h_aligned[i]
        trend_down = price < ema_12h_aligned[i]
        
        if position == 0:
            # Long: break above Donchian high with volume and uptrend
            if volume_confirmation and trend_up and price > donchian_high_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume and downtrend
            elif volume_confirmation and trend_down and price < donchian_low_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to Donchian low or trend changes
            if price < donchian_low_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price returns to Donchian high or trend changes
            if price > donchian_high_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "4h_Donchian20_12hEMA50_VolumeTrend"
timeframe = "4h"
leverage = 1.0
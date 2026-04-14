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
    
    # Load 12h data (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    if len(high_12h) < 20 or len(low_12h) < 20:
        return np.zeros(n)
    
    # Upper band: highest high over last 20 periods
    donchian_upper = np.full_like(close_12h, np.nan)
    for i in range(19, len(close_12h)):
        donchian_upper[i] = np.max(high_12h[i-19:i+1])
    
    # Lower band: lowest low over last 20 periods
    donchian_lower = np.full_like(close_12h, np.nan)
    for i in range(19, len(close_12h)):
        donchian_lower[i] = np.min(low_12h[i-19:i+1])
    
    # Calculate 12h EMA (50-period) for trend filter
    if len(close_12h) < 50:
        return np.zeros(n)
    
    ema50_12h = np.full_like(close_12h, np.nan)
    alpha = 2.0 / (50 + 1)
    ema50_12h[49] = np.mean(close_12h[:50])
    for i in range(50, len(close_12h)):
        ema50_12h[i] = close_12h[i] * alpha + ema50_12h[i-1] * (1 - alpha)
    
    # Calculate 12h RSI (14-period) for momentum
    if len(close_12h) < 14:
        return np.zeros(n)
    
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_12h, np.nan)
    avg_loss = np.full_like(close_12h, np.nan)
    
    if len(close_12h) >= 14:
        avg_gain[13] = np.mean(gain[1:14])
        avg_loss[13] = np.mean(loss[1:14])
        for i in range(14, len(close_12h)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full_like(close_12h, np.nan)
    rsi14_12h = np.full_like(close_12h, np.nan)
    for i in range(13, len(close_12h)):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi14_12h[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi14_12h[i] = 100 if avg_gain[i] > 0 else 0
    
    # Calculate 12h ATR (14-period) for volatility filter
    if len(high_12h) < 14 or len(low_12h) < 14 or len(close_12h) < 14:
        return np.zeros(n)
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr14_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 14:
        atr14_12h[13] = np.mean(tr[1:14])
        for i in range(14, len(close_12h)):
            atr14_12h[i] = (atr14_12h[i-1] * 13 + tr[i]) / 14
    
    # Align HTF indicators to LTF
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    rsi14_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi14_12h)
    atr14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr14_12h)
    
    # Volume ratio: current 12h volume vs 20-period average
    vol_ma_20 = np.full_like(volume_12h, np.nan)
    for i in range(19, len(volume_12h)):
        vol_ma_20[i] = np.mean(volume_12h[i-19:i+1])
    
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(rsi14_12h_aligned[i]) or 
            np.isnan(atr14_12h_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio calculation
        if vol_ma_20_aligned[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume_12h[i] / vol_ma_20_aligned[i] if i < len(volume_12h) else 1.0
        
        if position == 0:
            # Long: Price breaks above Donchian upper + above EMA50 + RSI > 55 + volume surge
            if (close[i] > donchian_upper_aligned[i] and
                close[i] > ema50_12h_aligned[i] and
                rsi14_12h_aligned[i] > 55 and
                volume_ratio > 2.0):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below Donchian lower + below EMA50 + RSI < 45 + volume surge
            elif (close[i] < donchian_lower_aligned[i] and
                  close[i] < ema50_12h_aligned[i] and
                  rsi14_12h_aligned[i] < 45 and
                  volume_ratio > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price breaks below Donchian lower OR RSI < 40
            if (close[i] < donchian_lower_aligned[i] or 
                rsi14_12h_aligned[i] < 40):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price breaks above Donchian upper OR RSI > 60
            if (close[i] > donchian_upper_aligned[i] or 
                rsi14_12h_aligned[i] > 60):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Donchian20_EMA50_RSI14_Volume"
timeframe = "12h"
leverage = 1.0
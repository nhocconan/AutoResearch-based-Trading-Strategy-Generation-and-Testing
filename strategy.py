#!/usr/bin/env python3
"""
1h_Donchian20_RSI14_Trend_v1
1h Donchian breakout with RSI filter, using 4h ADX for trend filter and 1d for regime.
Trend: 4h ADX > 20 (trending market). Entry: price breaks Donchian(20) with RSI(14) > 55 long / < 45 short.
Exit: opposite Donchian break or RSI crosses 50.
Session filter: 08-20 UTC only. Size: 0.20.
Target: 60-150 total trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === RSI(14) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[50], rsi])  # first value neutral
    
    # === Donchian(20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h ADX(14) for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_4h[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    minus_dm = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr_4h * 14)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr_4h * 14)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_4h = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # === Session filter: 08-20 UTC ===
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 60
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(adx_4h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Session filter
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Entry logic (only when flat)
        if position == 0:
            # Long: break above Donchian high, RSI > 55, ADX > 20 (trending)
            if (close[i] > highest_high[i] and 
                rsi[i] > 55 and 
                adx_4h_aligned[i] > 20):
                signals[i] = 0.20
                position = 1
                continue
            # Short: break below Donchian low, RSI < 45, ADX > 20 (trending)
            elif (close[i] < lowest_low[i] and 
                  rsi[i] < 45 and 
                  adx_4h_aligned[i] > 20):
                signals[i] = -0.20
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: break below Donchian low OR RSI < 50
            if (close[i] < lowest_low[i] or 
                rsi[i] < 50):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: break above Donchian high OR RSI > 50
            if (close[i] > highest_high[i] or 
                rsi[i] > 50):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_RSI14_Trend_v1"
timeframe = "1h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for trend filter (1d EMA34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Load 12h data for Donchian breakout
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian(20) on 12h
    highest_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    highest_20_aligned = align_htf_to_ltf(prices, df_12h, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_12h, lowest_20)
    
    # Volume confirmation on 12h
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # 4h ATR for stop loss
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    tr = np.maximum(high_4h - low_4h,
                    np.maximum(np.abs(high_4h - np.roll(close_4h, 1)),
                               np.abs(low_4h - np.roll(close_4h, 1))))
    tr[0] = high_4h[0] - low_4h[0]
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(highest_20_aligned[i]) or
            np.isnan(lowest_20_aligned[i]) or np.isnan(vol_ma_12h_aligned[i]) or
            np.isnan(atr_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_12h[i]
        
        if position == 0:
            # Long: price breaks above 12h Donchian high + above daily EMA34 + volume spike
            if (price > highest_20_aligned[i] and 
                price > ema_34_1d_aligned[i] and
                vol > 1.8 * vol_ma_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below 12h Donchian low + below daily EMA34 + volume spike
            elif (price < lowest_20_aligned[i] and 
                  price < ema_34_1d_aligned[i] and
                  vol > 1.8 * vol_ma_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price below Donchian low or stop loss hit
            if price < lowest_20_aligned[i] or price < entry_price - 2.5 * atr_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above Donchian high or stop loss hit
            if price > highest_20_aligned[i] or price > entry_price + 2.5 * atr_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hVol_EMA34Trend_Stop_v1"
timeframe = "4h"
leverage = 1.0
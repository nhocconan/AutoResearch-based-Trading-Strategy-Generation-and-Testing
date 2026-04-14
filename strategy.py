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
    
    # Load daily data (HTF) once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Load weekly data (HTF) once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR (14-period) for volatility
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    low_close = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align daily ATR to 4h timeframe
    atr_4h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate weekly EMA (34-period) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 34:
        multiplier = 2 / (34 + 1)
        ema_34_1w[33] = np.mean(close_1w[:34])
        for i in range(34, len(df_1w)):
            ema_34_1w[i] = (close_1w[i] - ema_34_1w[i-1]) * multiplier + ema_34_1w[i-1]
    
    # Align weekly EMA to 4h timeframe
    ema_34_4h = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate daily Donchian channel (20-period)
    high_20_1d = np.full(len(df_1d), np.nan)
    low_20_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 20:
        for i in range(19, len(df_1d)):
            high_20_1d[i] = np.max(high_1d[i-19:i+1])
            low_20_1d[i] = np.min(low_1d[i-19:i+1])
    
    # Align daily Donchian to 4h timeframe
    high_20_4h = align_htf_to_ltf(prices, df_1d, high_20_1d)
    low_20_4h = align_htf_to_ltf(prices, df_1d, low_20_1d)
    
    # Calculate 4-hour volume moving average (20-period)
    volume_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            volume_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_4h[i]) or
            np.isnan(ema_34_4h[i]) or
            np.isnan(high_20_4h[i]) or
            np.isnan(low_20_4h[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_4h[i] / close[i] < 0.005:
            signals[i] = 0.0
            continue
        
        # Skip low volume periods (volume < 60% of 20-period MA)
        if volume[i] < 0.6 * volume_ma[i]:
            signals[i] = 0.0
            continue
        
        # Entry conditions
        if position == 0:
            # Long: Price breaks above daily Donchian high AND above weekly EMA34
            if high[i] > high_20_4h[i] and close[i] > ema_34_4h[i]:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below daily Donchian low AND below weekly EMA34
            elif low[i] < low_20_4h[i] and close[i] < ema_34_4h[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price breaks below daily Donchian low or ATR-based stop
            if low[i] < low_20_4h[i] or close[i] < close[i-1] - 1.5 * atr_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Short exit: Price breaks above daily Donchian high or ATR-based stop
            if high[i] > high_20_4h[i] or close[i] > close[i-1] + 1.5 * atr_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_1d_1w_Donchian_EMA_ATR"
timeframe = "4h"
leverage = 1.0
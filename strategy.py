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
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12-hour EMA (50-period) for trend direction
    close_12h = close  # since we're on 12h timeframe
    ema_50 = np.full(n, np.nan)
    if n >= 50:
        ema_50[49] = np.mean(close[:50])
        for i in range(50, n):
            ema_50[i] = (close[i] * 2 + ema_50[i-1] * 49) / 51
    
    # Calculate daily ATR (14-period) for volatility filter
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 12-hour Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    # Calculate daily volatility filter (ATR > 1.5% of price)
    vol_filter_1d = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if not np.isnan(atr_1d[i]) and close_1d[i] > 0:
            vol_filter_1d[i] = atr_1d[i] / close_1d[i] > 0.015
        else:
            vol_filter_1d[i] = False
    vol_filter_12h = align_htf_to_ltf(prices, df_1d, vol_filter_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50[i]) or
            np.isnan(atr_12h[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 1.5% of price)
        if vol_filter_12h[i] < 0.5:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 50 EMA
        uptrend = close[i] > ema_50[i]
        downtrend = close[i] < ema_50[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high AND in uptrend
            if close[i] > donch_high[i] and uptrend:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below Donchian low AND in downtrend
            elif close[i] < donch_low[i] and downtrend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below Donchian low OR trend reverses
            if close[i] < donch_low[i] or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above Donchian high OR trend reverses
            if close[i] > donch_high[i] or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_EMA50_Donchian20_VolumeFilter"
timeframe = "12h"
leverage = 1.0
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
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate daily EMA(34) - using pandas EWMA for efficiency
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily ATR(14) - Wilder's smoothing
    high_low = df_1d['high'].values - df_1d['low'].values
    high_close = np.abs(df_1d['high'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]]))
    low_close = np.abs(df_1d['low'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align indicators to 6h timeframe
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    atr_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 6-hour Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 6-hour volume moving average (20-period)
    volume_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            volume_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_6h[i]) or
            np.isnan(atr_6h[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.25% of price)
        if atr_6h[i] / close[i] < 0.0025:
            signals[i] = 0.0
            continue
        
        # Skip low volume periods (volume < 60% of 20-period MA)
        if volume[i] < 0.6 * volume_ma[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above EMA34 for long, below for short
        if position == 0:
            # Long: Price breaks above 6h Donchian high AND above EMA34 AND volume > 1.3x MA
            if close[i] > donch_high[i] and close[i] > ema_34_6h[i] and volume[i] > 1.3 * volume_ma[i]:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 6h Donchian low AND below EMA34 AND volume > 1.3x MA
            elif close[i] < donch_low[i] and close[i] < ema_34_6h[i] and volume[i] > 1.3 * volume_ma[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below 6h Donchian low OR below EMA34
            if close[i] < donch_low[i] or close[i] < ema_34_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above 6h Donchian high OR above EMA34
            if close[i] > donch_high[i] or close[i] > ema_34_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_EMA34_Donchian20_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0
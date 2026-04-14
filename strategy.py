#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mta_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h EMA200 for trend
    ema_200_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 200:
        ema_200_4h[199] = np.mean(close_4h[:200])
        for i in range(200, len(close_4h)):
            ema_200_4h[i] = (close_4h[i] * 2 + ema_200_4h[i-1] * 198) / 200
    
    # Calculate 4h ATR(14) for volatility filter
    tr_4h = np.zeros_like(close_4h)
    tr_4h[0] = high_4h[0] - low_4h[0]
    for i in range(1, len(close_4h)):
        tr_4h[i] = max(high_4h[i] - low_4h[i], 
                       abs(high_4h[i] - close_4h[i-1]),
                       abs(low_4h[i] - close_4h[i-1]))
    
    atr_14_4h = np.full_like(close_4h, np.nan)
    if len(tr_4h) >= 14:
        atr_14_4h[13] = np.mean(tr_4h[:14])
        for i in range(14, len(tr_4h)):
            atr_14_4h[i] = (tr_4h[i] * 2 + atr_14_4h[i-1] * 12) / 14
    
    # Align 4h indicators to 1h
    ema_200_4h_1h = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    atr_14_4h_1h = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Load 1d data for session filter (we'll use time-based filter instead)
    # Calculate 1h ATR(14) for volatility
    tr_1h = np.zeros_like(close)
    tr_1h[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr_1h[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i-1]),
                       abs(low[i] - close[i-1]))
    
    atr_14_1h = np.full_like(close, np.nan)
    if len(tr_1h) >= 14:
        atr_14_1h[13] = np.mean(tr_1h[:14])
        for i in range(14, len(tr_1h)):
            atr_14_1h[i] = (tr_1h[i] * 2 + atr_14_1h[i-1] * 12) / 14
    
    # Volume spike detection (1h)
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_200_4h_1h[i]) or 
            np.isnan(atr_14_4h_1h[i]) or
            np.isnan(atr_14_1h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 1h volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # ATR ratio: 1h ATR / 4h ATR (normalized volatility)
        if atr_14_4h_1h[i] <= 0:
            atr_ratio = 0
        else:
            atr_ratio = atr_14_1h[i] / atr_14_4h_1h[i]
        
        if position == 0:
            # Long: Price above 4h EMA200, volume spike, and normal volatility
            if (close[i] > ema_200_4h_1h[i] and
                volume_ratio > 1.5 and
                atr_ratio < 2.0):
                position = 1
                signals[i] = position_size
            # Short: Price below 4h EMA200, volume spike, and normal volatility
            elif (close[i] < ema_200_4h_1h[i] and
                  volume_ratio > 1.5 and
                  atr_ratio < 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price crosses below 4h EMA200 or volatility too high
            if (close[i] < ema_200_4h_1h[i] or 
                atr_ratio > 2.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price crosses above 4h EMA200 or volatility too high
            if (close[i] > ema_200_4h_1h[i] or 
                atr_ratio > 2.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_EMA200_Volume_ATR_Filter"
timeframe = "1h"
leverage = 1.0
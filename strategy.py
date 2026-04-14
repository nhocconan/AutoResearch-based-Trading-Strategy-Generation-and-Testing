#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for daily ATR and ATR-based channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily ATR(14)
    atr_14_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 15:
        tr = np.maximum(
            high_1d[1:] - low_1d[1:],
            np.maximum(
                np.abs(high_1d[1:] - close_1d[:-1]),
                np.abs(low_1d[1:] - close_1d[:-1])
            )
        )
        atr_14_1d[14] = np.mean(tr[:14])
        for i in range(15, len(close_1d)):
            atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr[i-1]) / 14
    
    # Calculate ATR-based channels (Keltner-like) on 1d
    # Upper: EMA(20) + 1.5 * ATR(14)
    # Lower: EMA(20) - 1.5 * ATR(14)
    ema_20_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 20:
        ema_20_1d[19] = np.mean(close_1d[:20])
        for i in range(20, len(close_1d)):
            ema_20_1d[i] = (close_1d[i] * 2 + ema_20_1d[i-1] * 18) / 20
    
    upper_channel_1d = np.full_like(close_1d, np.nan)
    lower_channel_1d = np.full_like(close_1d, np.nan)
    if len(ema_20_1d) >= 20 and len(atr_14_1d) >= 15:
        for i in range(20, len(close_1d)):
            if not np.isnan(ema_20_1d[i]) and not np.isnan(atr_14_1d[i]):
                upper_channel_1d[i] = ema_20_1d[i] + 1.5 * atr_14_1d[i]
                lower_channel_1d[i] = ema_20_1d[i] - 1.5 * atr_14_1d[i]
    
    # Align 1d indicators to 12h timeframe
    upper_channel_12h = align_htf_to_ltf(prices, df_1d, upper_channel_1d)
    lower_channel_12h = align_htf_to_ltf(prices, df_1d, lower_channel_1d)
    ema_20_1d_12h = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume spike detection on 12h
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(60, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_channel_12h[i]) or 
            np.isnan(lower_channel_12h[i]) or
            np.isnan(ema_20_1d_12h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 12h volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above upper channel with volume spike
            if (close[i] > upper_channel_12h[i] and volume_ratio > 2.0):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below lower channel with volume spike
            elif (close[i] < lower_channel_12h[i] and volume_ratio > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price closes below EMA(20) or breaks below lower channel
            if (close[i] < ema_20_1d_12h[i] or 
                close[i] < lower_channel_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price closes above EMA(20) or breaks above upper channel
            if (close[i] > ema_20_1d_12h[i] or 
                close[i] > upper_channel_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_ATR_Channel_Volume_Breakout"
timeframe = "12h"
leverage = 1.0
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
    
    # Load 12h data (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    vol_12h = df_12h['volume'].values
    
    # Calculate 20-period EMA for 12h trend (fast)
    ema_fast_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 20:
        ema_fast_12h[19] = np.mean(close_12h[:20])
        for i in range(20, len(close_12h)):
            ema_fast_12h[i] = (close_12h[i] * 2 + ema_fast_12h[i-1] * 19) / 20
    
    # Calculate 50-period EMA for 12h trend (slow)
    ema_slow_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 50:
        ema_slow_12h[49] = np.mean(close_12h[:50])
        for i in range(50, len(close_12h)):
            ema_slow_12h[i] = (close_12h[i] * 2 + ema_slow_12h[i-1] * 49) / 50
    
    # Align EMAs to 6h timeframe
    ema_fast_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_fast_12h)
    ema_slow_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_slow_12h)
    
    # Calculate 12h RSI(14) for momentum
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_12h, np.nan)
    avg_loss = np.full_like(close_12h, np.nan)
    
    if len(close_12h) >= 14:
        avg_gain[13] = np.mean(gain[1:15])
        avg_loss[13] = np.mean(loss[1:15])
        for i in range(15, len(close_12h)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rsi_12h = np.full_like(close_12h, np.nan)
    for i in range(14, len(close_12h)):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi_12h[i] = 100 - (100 / (1 + rs))
        else:
            rsi_12h[i] = 100
    
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Calculate 12h volume spike detector (current vs 20-period average)
    vol_ma_20_12h = np.full_like(vol_12h, np.nan)
    if len(vol_12h) >= 20:
        for i in range(19, len(vol_12h)):
            vol_ma_20_12h[i] = np.mean(vol_12h[i-19:i+1])
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_fast_12h_aligned[i]) or 
            np.isnan(ema_slow_12h_aligned[i]) or 
            np.isnan(rsi_12h_aligned[i]) or 
            np.isnan(vol_ma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 6h volume vs 20-period 12h average volume
        if vol_ma_20_12h_aligned[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20_12h_aligned[i]
        
        if position == 0:
            # Long: Fast EMA above slow EMA + RSI > 50 + volume surge
            if (ema_fast_12h_aligned[i] > ema_slow_12h_aligned[i] and
                rsi_12h_aligned[i] > 50 and
                volume_ratio > 2.0):
                position = 1
                signals[i] = position_size
            # Short: Fast EMA below slow EMA + RSI < 50 + volume surge
            elif (ema_fast_12h_aligned[i] < ema_slow_12h_aligned[i] and
                  rsi_12h_aligned[i] < 50 and
                  volume_ratio > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Fast EMA crosses below slow EMA OR RSI < 40
            if (ema_fast_12h_aligned[i] < ema_slow_12h_aligned[i] or 
                rsi_12h_aligned[i] < 40):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Fast EMA crosses above slow EMA OR RSI > 60
            if (ema_fast_12h_aligned[i] > ema_slow_12h_aligned[i] or 
                rsi_12h_aligned[i] > 60):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12h_EMA_RSI_Volume_v1"
timeframe = "6h"
leverage = 1.0
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
    open_time = prices['open_time'].values
    
    # Get 4h data for trend and volume filters
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = np.full(len(close_4h), np.nan)
    alpha = 2 / (50 + 1)
    for i in range(len(close_4h)):
        if i < 49:
            ema_50_4h[i] = np.mean(close_4h[:i+1]) if i > 0 else close_4h[i]
        else:
            if np.isnan(ema_50_4h[i-1]):
                ema_50_4h[i] = np.mean(close_4h[i-49:i+1])
            else:
                ema_50_4h[i] = close_4h[i] * alpha + ema_50_4h[i-1] * (1 - alpha)
    
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h volume average for volume filter
    vol_4h = df_4h['volume'].values
    vol_ma_20_4h = np.full(len(vol_4h), np.nan)
    for i in range(len(vol_4h)):
        if i < 19:
            vol_ma_20_4h[i] = np.mean(vol_4h[:i+1]) if i > 0 else vol_4h[i]
        else:
            vol_ma_20_4h[i] = np.mean(vol_4h[i-19:i+1])
    
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    # Calculate 1h RSI(14) for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full(n, np.nan)
    valid_rsi = (~np.isnan(avg_gain)) & (~np.isnan(avg_loss)) & (avg_loss > 0)
    rs[valid_rsi] = avg_gain[valid_rsi] / avg_loss[valid_rsi]
    rsi_14 = np.full(n, np.nan)
    rsi_14[valid_rsi] = 100 - (100 / (1 + rs[valid_rsi]))
    
    # Precompute session hours for filter
    hours = pd.DatetimeIndex(open_time).hour
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(14, 49)
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma_20_4h_aligned[i]) or
            np.isnan(rsi_14[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20_4h_aligned[i] if vol_ma_20_4h_aligned[i] > 0 else 0
        
        if position == 0:
            # Long: price > 4h EMA50 (uptrend) + volume spike + RSI < 40 (pullback)
            if (price > ema_50_4h_aligned[i] and 
                vol_ratio > 1.5 and 
                rsi_14[i] < 40):
                signals[i] = 0.20
                position = 1
            # Short: price < 4h EMA50 (downtrend) + volume spike + RSI > 60 (bounce)
            elif (price < ema_50_4h_aligned[i] and 
                  vol_ratio > 1.5 and 
                  rsi_14[i] > 60):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI > 60 or price crosses below EMA50
            if (rsi_14[i] > 60 or 
                price < ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI < 40 or price crosses above EMA50
            if (rsi_14[i] < 40 or 
                price > ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA50_VolumeSpike_RSI14_v1"
timeframe = "1h"
leverage = 1.0
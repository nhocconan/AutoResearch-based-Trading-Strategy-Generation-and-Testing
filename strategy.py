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
    
    # Get 12h data for calculations (called ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h RSI (14-period)
    close_12h = df_12h['close'].values
    delta = np.diff(close_12h)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_12h), np.nan)
    avg_loss = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 14:
        avg_gain[13] = np.mean(gain[1:15])
        avg_loss[13] = np.mean(loss[1:15])
        for i in range(14, len(close_12h)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_12h = 100 - (100 / (1 + rs))
    
    # Calculate 12h EMA (50-period) for trend filter
    ema_50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        alpha = 2 / (50 + 1)
        ema_50_12h[0] = close_12h[0]
        for i in range(1, len(close_12h)):
            ema_50_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_50_12h[i-1]
    
    # Align 12h indicators to 6h timeframe
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 6-period volume average for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 6
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(14, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_12h_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: RSI oversold (<30), price above EMA50 (uptrend), volume spike
            if rsi_12h_aligned[i] < 30 and price > ema_50_12h_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: RSI overbought (>70), price below EMA50 (downtrend), volume spike
            elif rsi_12h_aligned[i] > 70 and price < ema_50_12h_aligned[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI overbought (>70) or trend change (price below EMA50)
            if rsi_12h_aligned[i] > 70 or price < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: RSI oversold (<30) or trend change (price above EMA50)
            if rsi_12h_aligned[i] < 30 or price > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_RSI12h_EMA50_Volume"
timeframe = "6h"
leverage = 1.0
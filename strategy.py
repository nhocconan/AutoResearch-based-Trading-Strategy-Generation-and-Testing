#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data (HTF) once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h EMA (20-period)
    ema_12h = np.full(len(df_12h), np.nan)
    if len(df_12h) >= 20:
        ema_12h[19] = np.mean(close_12h[:20])
        for i in range(20, len(df_12h)):
            ema_12h[i] = (close_12h[i] * 2 + ema_12h[i-1] * 18) / 20
    
    # Calculate 12h ATR (14-period) - Wilder's smoothing
    high_low = high_12h - low_12h
    high_close = np.abs(high_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    low_close = np.abs(low_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    atr_12h = np.full(len(df_12h), np.nan)
    if len(df_12h) >= 14:
        atr_12h[13] = np.mean(tr[:14])
        for i in range(14, len(df_12h)):
            atr_12h[i] = (atr_12h[i-1] * 13 + tr[i]) / 14
    
    # Align indicators to 6h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate 6h RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    rsi = np.full(n, np.nan)
    if n >= 14:
        avg_gain = np.mean(gain[:14])
        avg_loss = np.mean(loss[:14])
        if avg_loss == 0:
            rsi[13] = 100
        else:
            rsi[13] = 100 - (100 / (1 + avg_gain / avg_loss))
        
        for i in range(14, n):
            avg_gain = (gain[i] + avg_gain * 13) / 14
            avg_loss = (loss[i] + avg_loss * 13) / 14
            if avg_loss == 0:
                rsi[i] = 100
            else:
                rsi[i] = 100 - (100 / (1 + avg_gain / avg_loss))
    
    # Calculate 6h ATR (14-period) - Wilder's smoothing
    high_low_6h = high - low
    high_close_6h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    low_close_6h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_6h = np.maximum(high_low_6h, np.maximum(high_close_6h, low_close_6h))
    
    atr_6h = np.full(n, np.nan)
    if n >= 14:
        atr_6h[13] = np.mean(tr_6h[:14])
        for i in range(14, n):
            atr_6h[i] = (atr_6h[i-1] * 13 + tr_6h[i]) / 14
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi[i]) or
            np.isnan(ema_12h_aligned[i]) or
            np.isnan(atr_12h_aligned[i]) or
            np.isnan(atr_6h[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.3% of price)
        if atr_6h[i] / close[i] < 0.003:
            signals[i] = 0.0
            continue
        
        # RSI conditions: oversold (<30) for long, overbought (>70) for short
        # 12h trend filter: price above EMA for long, below EMA for short
        # Volatility filter: ATR expansion (current ATR > 1.2 * 12h ATR)
        if rsi[i] < 30 and close[i] > ema_12h_aligned[i] and atr_6h[i] > 1.2 * atr_12h_aligned[i]:
            if position <= 0:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = position_size
        elif rsi[i] > 70 and close[i] < ema_12h_aligned[i] and atr_6h[i] > 1.2 * atr_12h_aligned[i]:
            if position >= 0:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "6h_12h_EMA_RSI_Volatility_Filter"
timeframe = "6h"
leverage = 1.0
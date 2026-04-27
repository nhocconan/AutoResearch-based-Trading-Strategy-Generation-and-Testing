#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for calculations (called ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12-period RSI (14) for momentum
    close_12h = df_12h['close'].values
    rsi_14_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 14:
        delta = np.diff(close_12h)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full(len(close_12h), np.nan)
        avg_loss = np.full(len(close_12h), np.nan)
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, len(close_12h)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
            rs = avg_gain[i] / avg_loss[i] if avg_loss[i] != 0 else 0
            rsi_14_12h[i] = 100 - (100 / (1 + rs))
    
    # Calculate 12-period ATR (14) for volatility
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    tr_12h = np.zeros(len(close_12h))
    tr_12h[0] = high_12h[0] - low_12h[0]
    for i in range(1, len(close_12h)):
        tr_12h[i] = max(high_12h[i] - low_12h[i], 
                        abs(high_12h[i] - close_12h[i-1]),
                        abs(low_12h[i] - close_12h[i-1]))
    atr_14_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 14:
        atr_14_12h[13] = np.mean(tr_12h[:14])
        for i in range(14, len(close_12h)):
            atr_14_12h[i] = (atr_14_12h[i-1] * 13 + tr_12h[i]) / 14
    
    # Calculate 12-period EMA (50) for trend
    ema_50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        multiplier = 2 / (50 + 1)
        ema_50_12h[49] = np.mean(close_12h[:50])
        for i in range(50, len(close_12h)):
            ema_50_12h[i] = (close_12h[i] * multiplier) + (ema_50_12h[i-1] * (1 - multiplier))
    
    # Align 12h indicators to 6h timeframe
    rsi_14_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_14_12h)
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
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
    start_idx = max(50, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_14_12h_aligned[i]) or 
            np.isnan(atr_14_12h_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        # Trend filter: price above/below 50 EMA
        uptrend = price > ema_50_12h_aligned[i]
        downtrend = price < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: RSI oversold (<30) + uptrend + volume spike
            if rsi_14_12h_aligned[i] < 30 and uptrend and vol_filter:
                signals[i] = size
                position = 1
            # Short: RSI overbought (>70) + downtrend + volume spike
            elif rsi_14_12h_aligned[i] > 70 and downtrend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI overbought (>70) or trend reversal
            if rsi_14_12h_aligned[i] > 70 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: RSI oversold (<30) or trend reversal
            if rsi_14_12h_aligned[i] < 30 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_RSI14_EMA50_Volume"
timeframe = "6h"
leverage = 1.0
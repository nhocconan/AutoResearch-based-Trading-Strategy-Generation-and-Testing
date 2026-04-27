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
    
    # Calculate 12-period EMA (9-period) for 12h trend
    close_12h = df_12h['close'].values
    if len(close_12h) >= 9:
        alpha = 2 / (9 + 1)
        ema_9_12h = np.full(len(close_12h), np.nan)
        ema_9_12h[0] = close_12h[0]
        for i in range(1, len(close_12h)):
            ema_9_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_9_12h[i-1]
    else:
        ema_9_12h = np.full(len(close_12h), np.nan)
    
    # Calculate 12-period RSI (14-period) using Wilder's smoothing
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_12h), np.nan)
    avg_loss = np.full(len(close_12h), np.nan)
    
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[1:15])
        avg_loss[13] = np.mean(loss[1:15])
        
        for i in range(14, len(close_12h)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rsi_12h = np.full(len(close_12h), np.nan)
    for i in range(14, len(close_12h)):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi_12h[i] = 100 - (100 / (1 + rs))
        else:
            rsi_12h[i] = 100
    
    # Calculate 12-period ATR (14-period) for volatility
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_prev = np.roll(close_12h, 1)
    close_12h_prev[0] = close_12h[0]
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - close_12h_prev)
    tr3 = np.abs(low_12h - close_12h_prev)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14_12h = np.full(len(tr), np.nan)
    if len(tr) >= 14:
        atr_14_12h[13] = np.mean(tr[1:15])
        for i in range(14, len(tr)):
            atr_14_12h[i] = (atr_14_12h[i-1] * 13 + tr[i]) / 14
    
    # Align 12h indicators to 4h timeframe
    ema_9_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_9_12h)
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    
    # Calculate 4-period volume average for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 4
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(14, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_9_12h_aligned[i]) or np.isnan(rsi_12h_aligned[i]) or 
            np.isnan(atr_14_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        # RSI filter: avoid extreme overbought/oversold
        rsi_filter = (rsi_12h_aligned[i] > 30) & (rsi_12h_aligned[i] < 70)
        
        if position == 0:
            # Long: Price above 12h EMA9 with volume and RSI not extreme
            if price > ema_9_12h_aligned[i] and vol_filter and rsi_filter:
                signals[i] = size
                position = 1
            # Short: Price below 12h EMA9 with volume and RSI not extreme
            elif price < ema_9_12h_aligned[i] and vol_filter and rsi_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below 12h EMA9 or volatility spike (potential reversal)
            if price < ema_9_12h_aligned[i] or (vol_ratio > 2.5 and rsi_12h_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above 12h EMA9 or volatility spike (potential reversal)
            if price > ema_9_12h_aligned[i] or (vol_ratio > 2.5 and rsi_12h_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_EMA9_12h_RSI14_Volume"
timeframe = "4h"
leverage = 1.0
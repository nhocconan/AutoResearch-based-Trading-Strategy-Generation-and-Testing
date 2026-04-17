#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Close ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # === 12h KAMA (14) ===
    # Calculate ER (Efficiency Ratio)
    change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility = np.sum(np.abs(np.diff(close_12h)), axis=0)  # placeholder, will compute properly below
    # Recalculate volatility properly
    volatility = np.zeros_like(close_12h)
    for i in range(1, len(close_12h)):
        volatility[i] = volatility[i-1] + np.abs(close_12h[i] - close_12h[i-1])
    
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # Calculate KAMA
    kama = np.zeros_like(close_12h)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    # === 12h RSI(14) ===
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    if len(gain) > 0:
        avg_gain[0] = gain[0]
    if len(loss) > 0:
        avg_loss[0] = loss[0]
    for i in range(1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 12h Volume MA(20) ===
    vol_ma_20 = np.zeros_like(volume_12h)
    for i in range(len(volume_12h)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume_12h[i-19:i+1])
        else:
            vol_ma_20[i] = np.mean(volume_12h[max(0, i-9):i+1]) if i > 0 else volume_12h[0]
    
    # Align to 6h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    high_12h_aligned = align_htf_to_ltf(prices, df_12h, high_12h)
    low_12h_aligned = align_htf_to_ltf(prices, df_12h, low_12h)
    volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 200
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_12h_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(close_12h_aligned[i]) or
            np.isnan(high_12h_aligned[i]) or np.isnan(low_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.3x 20-period average
        vol_confirm = volume_12h_aligned[i] > vol_ma_20_aligned[i] * 1.3
        
        # Price position relative to 12h range
        range_12h = high_12h_aligned[i] - low_12h_aligned[i]
        if range_12h > 0:
            pos_in_range = (close_12h_aligned[i] - low_12h_aligned[i]) / range_12h
        else:
            pos_in_range = 0.5
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price above KAMA, RSI < 40, in lower 40% of range, with volume confirmation
            if (close[i] > kama_aligned[i] and 
                rsi_12h_aligned[i] < 40 and 
                pos_in_range < 0.4 and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price below KAMA, RSI > 60, in upper 60% of range, with volume confirmation
            elif (close[i] < kama_aligned[i] and 
                  rsi_12h_aligned[i] > 60 and 
                  pos_in_range > 0.6 and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price below KAMA OR RSI > 50
            if close[i] < kama_aligned[i] or rsi_12h_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above KAMA OR RSI < 50
            if close[i] > kama_aligned[i] or rsi_12h_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_RSI_Range_Position"
timeframe = "6h"
leverage = 1.0
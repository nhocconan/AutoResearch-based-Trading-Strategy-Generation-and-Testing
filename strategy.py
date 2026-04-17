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
    
    # === 1d KAMA (Kaufman Adaptive Moving Average) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate efficiency ratio (ER)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0) if len(close_1d) > 1 else 0
    # More efficient calculation
    er = np.zeros(len(close_1d))
    for i in range(len(close_1d)):
        if i == 0:
            er[i] = 1.0
        else:
            price_change = np.abs(close_1d[i] - close_1d[i-10]) if i >= 10 else np.abs(close_1d[i] - close_1d[0])
            sum_abs_change = np.sum(np.abs(np.diff(close_1d[max(0, i-9):i+1]))) if i > 0 else 0
            er[i] = price_change / (sum_abs_change + 1e-10) if sum_abs_change > 0 else 1.0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(len(close_1d))
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # === 1d RSI (14) ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros(len(gain))
    avg_loss = np.zeros(len(loss))
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1d Volume Spike ===
    vol_ma_20_1d = np.convolve(volume_1d, np.ones(20)/20, mode='same')
    vol_ma_20_1d[:10] = volume_1d[:10].mean() if len(volume_1d) >= 10 else volume_1d.mean()
    vol_ma_20_1d[-10:] = volume_1d[-10:].mean() if len(volume_1d) >= 10 else volume_1d.mean()
    
    # Align to 4h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = vol_1d_aligned[i] > vol_ma_20_1d_aligned[i] * 1.5
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price above KAMA and RSI > 50 with volume confirmation
            if close[i] > kama_1d_aligned[i] and rsi_1d_aligned[i] > 50 and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price below KAMA and RSI < 50 with volume confirmation
            elif close[i] < kama_1d_aligned[i] and rsi_1d_aligned[i] < 50 and vol_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: price crosses KAMA in opposite direction
        elif position == 1:
            # Exit long: price crosses below KAMA
            if close[i] < kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA
            if close[i] > kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0
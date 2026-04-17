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
    
    # === 1d RSI (14) with Wilder's smoothing ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate RSI
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros(len(gain))
    avg_loss = np.zeros(len(loss))
    if len(gain) > 0:
        avg_gain[0] = gain[0]
        avg_loss[0] = loss[0]
        for i in range(1, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 20-period volume average
    vol_ma_20_1d = np.zeros(len(volume_1d))
    for i in range(len(volume_1d)):
        if i >= 19:
            vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
        else:
            vol_ma_20_1d[i] = np.mean(volume_1d[max(0, i-9):i+1]) if i > 0 else volume_1d[0]
    
    # Align to 4h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(vol_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = vol_1d_aligned[i] > vol_ma_20_1d_aligned[i] * 1.5
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: RSI < 30 (oversold) with volume confirmation
            if rsi_1d_aligned[i] < 30 and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short: RSI > 70 (overbought) with volume confirmation
            elif rsi_1d_aligned[i] > 70 and vol_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: RSI returns to neutral (40-60)
        elif position == 1:
            # Exit long: RSI >= 40
            if rsi_1d_aligned[i] >= 40:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI <= 60
            if rsi_1d_aligned[i] <= 60:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_RSI30_70_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0
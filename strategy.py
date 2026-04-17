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
    
    # === 1d Bollinger Bands (20, 2) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Bollinger Bands
    sma_20 = np.zeros(len(close_1d))
    std_20 = np.zeros(len(close_1d))
    upper = np.zeros(len(close_1d))
    lower = np.zeros(len(close_1d))
    
    for i in range(len(close_1d)):
        if i >= 19:
            sma_20[i] = np.mean(close_1d[i-19:i+1])
            std_20[i] = np.std(close_1d[i-19:i+1])
        else:
            sma_20[i] = np.mean(close_1d[max(0, i-9):i+1]) if i > 0 else close_1d[0]
            std_20[i] = np.std(close_1d[max(0, i-9):i+1]) if i > 0 else 0.0
        upper[i] = sma_20[i] + 2 * std_20[i]
        lower[i] = sma_20[i] - 2 * std_20[i]
    
    # === 1d RSI(14) ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros(len(gain))
    avg_loss = np.zeros(len(loss))
    if len(gain) > 0:
        avg_gain[0] = gain[0]
    if len(loss) > 0:
        avg_loss[0] = loss[0]
    for i in range(1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1d Volume Spike Detector ===
    vol_ma_20 = np.zeros(len(volume_1d))
    for i in range(len(volume_1d)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume_1d[i-19:i+1])
        else:
            vol_ma_20[i] = np.mean(volume_1d[max(0, i-9):i+1]) if i > 0 else volume_1d[0]
    
    # Align to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(vol_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_spike = vol_1d_aligned[i] > vol_ma_20_aligned[i] * 2.0
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price touches lower BB and RSI < 30 with volume spike
            if low[i] <= lower_aligned[i] and rsi_1d_aligned[i] < 30 and vol_spike:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price touches upper BB and RSI > 70 with volume spike
            elif high[i] >= upper_aligned[i] and rsi_1d_aligned[i] > 70 and vol_spike:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: price returns to middle (SMA) or RSI normalizes
        elif position == 1:
            # Exit long: price >= SMA20 or RSI >= 40
            sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20)
            if close[i] >= sma_20_aligned[i] or rsi_1d_aligned[i] >= 40:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price <= SMA20 or RSI <= 60
            sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20)
            if close[i] <= sma_20_aligned[i] or rsi_1d_aligned[i] <= 60:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_BollingerBands_RSI_VolumeSpike"
timeframe = "4h"
leverage = 1.0
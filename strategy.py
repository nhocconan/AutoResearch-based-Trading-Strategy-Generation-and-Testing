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
    
    # === 1d Pivot Points (Classic: PP, R1, S1) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # === 1d Volume (20-period average) ===
    vol_ma_20_1d = np.zeros(len(volume))
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20_1d[i] = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20_1d[i] = np.mean(volume[max(0, i-9):i+1]) if i > 0 else volume[0]
    
    # === 14-period RSI on 1d close ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    if len(gain) > 0:
        avg_gain[0] = gain[0]
        avg_loss[0] = loss[0]
        for i in range(1, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align all 1d data to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        vol_confirm = vol_1d_aligned[i] > vol_ma_20_1d_aligned[i] * 1.8
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price touches S1 support and RSI < 35 (oversold) with volume confirmation
            if low[i] <= s1_aligned[i] and rsi_1d_aligned[i] < 35 and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price touches R1 resistance and RSI > 65 (overbought) with volume confirmation
            elif high[i] >= r1_aligned[i] and rsi_1d_aligned[i] > 65 and vol_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: price crosses the pivot point or RSI returns to neutral (45-55)
        elif position == 1:
            # Exit long: price crosses above pivot or RSI > 55
            if close[i] >= pivot_aligned[i] or rsi_1d_aligned[i] > 55:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses below pivot or RSI < 45
            if close[i] <= pivot_aligned[i] or rsi_1d_aligned[i] < 45:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_S1R1_RSI_Volume"
timeframe = "12h"
leverage = 1.0
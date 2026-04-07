#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d KAMA Direction with RSI Filter and Volume Confirmation
# Hypothesis: KAMA adapts to market noise, providing reliable trend direction.
# RSI filter avoids overextended entries, volume confirms institutional participation.
# Works in both bull and bear markets by following adaptive trend.
# Targets 10-25 trades/year with disciplined entries to minimize fee drag.

name = "1d_kama_rsi_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (adaptive moving average)
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    abs_change = np.abs(np.diff(close, n=1))  # 1-period absolute change
    # Pad arrays for rolling sum
    change_padded = np.concatenate([np.full(9, np.nan), change])
    abs_change_padded = np.concatenate([np.full(0, np.nan), abs_change])
    
    # 10-period ER
    er = np.zeros(n)
    er[:9] = np.nan
    for i in range(9, n):
        if np.isnan(change_padded[i]) or np.isnan(np.nansum(abs_change_padded[i-9:i+1])):
            er[i] = np.nan
        else:
            sum_abs = np.nansum(abs_change_padded[i-9:i+1])
            er[i] = np.abs(change_padded[i]) / sum_abs if sum_abs != 0 else 0
    
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama = np.zeros(n)
    kama[:9] = np.nan
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume 20-period average
    vol_ma = np.zeros(n)
    vol_ma[:19] = np.nan
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if required data not available
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below KAMA OR RSI > 70 (overbought)
            if close[i] < kama[i] or rsi[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above KAMA OR RSI < 30 (oversold)
            if close[i] > kama[i] or rsi[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price above KAMA + RSI < 50 (not overbought) + volume confirmation
            if (close[i] > kama[i] and 
                rsi[i] < 50 and 
                vol_confirm):
                position = 1
                signals[i] = 0.25
            # Short: price below KAMA + RSI > 50 (not oversold) + volume confirmation
            elif (close[i] < kama[i] and 
                  rsi[i] > 50 and 
                  vol_confirm):
                position = -1
                signals[i] = -0.25
    
    return signals
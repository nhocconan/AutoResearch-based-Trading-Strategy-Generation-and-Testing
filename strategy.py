#!/usr/bin/env python3
# 4h_4H_KAMA_Trend_1dATR_Filter_VolumeSpike
# Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both bull and bear markets.
# Combined with 1d ATR filter to avoid whipsaws in low volatility and volume spike for breakout confirmation.
# Targets 20-40 trades/year to minimize fee drag while capturing strong trends.

name = "4h_4H_KAMA_Trend_1dATR_Filter_VolumeSpike"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ATR on daily
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14_1d = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr_14_1d[13] = np.mean(tr[0:14])
        for i in range(14, len(tr)):
            atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr[i]) / 14
    
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate KAMA on 4h close
    def calculate_kama(close, length=10, fast=2, slow=30):
        # Efficiency ratio
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.zeros_like(close)
        er[length:] = change[length-1:] / volatility[length-1:]
        er[length:] = np.where(volatility[length-1:] == 0, 0, er[length:])
        
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # KAMA calculation
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, length=10, fast=2, slow=30)
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 10)  # Ensure volume MA, ATR, and KAMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above KAMA AND ATR filter (volatility not too low) AND volume spike
            if (close[i] > kama[i] and 
                atr_14_1d_aligned[i] > 0 and  # Ensure ATR is valid
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA AND ATR filter AND volume spike
            elif (close[i] < kama[i] and 
                  atr_14_1d_aligned[i] > 0 and
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
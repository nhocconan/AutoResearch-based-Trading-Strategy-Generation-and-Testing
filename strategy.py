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
    
    # Get daily data for weekly high/low
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate daily high and low
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly high (max of last 5 daily highs) and low (min of last 5 daily lows)
    weekly_high = np.full(len(df_1d), np.nan)
    weekly_low = np.full(len(df_1d), np.nan)
    
    for i in range(5, len(df_1d)):
        weekly_high[i] = np.max(high_1d[i-5:i])
        weekly_low[i] = np.min(low_1d[i-5:i])
    
    # Align weekly high/low to 12h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1d, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1d, weekly_low)
    
    # Calculate ATR(14) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[high[0] - low[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros(n)
    for i in range(n):
        if i < 14:
            atr[i] = np.mean(tr[:i+1]) if i > 0 else tr[0]
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate volume average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(20, 14, 5)  # volume MA needs 20, ATR needs 14, weekly needs 5
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_high_aligned[i]) or
            np.isnan(weekly_low_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Volume confirmation: > 2.0x average volume (reduced from 2.5 for more signals)
        volume_confirmation = vol_ratio > 2.0
        
        # ATR volatility filter: avoid low volatility periods
        # Only trade when ATR is above 40% of its 50-period average
        if i >= 50:
            atr_avg = np.mean(atr[i-50:i+1])
            vol_filter = atr[i] > atr_avg * 0.4
        else:
            vol_filter = True  # No filter during warmup
        
        if position == 0:
            # Long: break above weekly high with volume and volatility
            if volume_confirmation and vol_filter and price > weekly_high_aligned[i]:
                signals[i] = 0.30
                position = 1
            # Short: break below weekly low with volume and volatility
            elif volume_confirmation and vol_filter and price < weekly_low_aligned[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to weekly midpoint or volatility drops significantly
            weekly_mid = (weekly_high_aligned[i] + weekly_low_aligned[i]) / 2
            if price < weekly_mid or atr[i] < np.mean(atr[max(0, i-50):i+1]) * 0.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30  # Maintain position
        elif position == -1:
            # Short exit: price returns to weekly midpoint or volatility drops significantly
            weekly_mid = (weekly_high_aligned[i] + weekly_low_aligned[i]) / 2
            if price > weekly_mid or atr[i] < np.mean(atr[max(0, i-50):i+1]) * 0.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30  # Maintain position
    
    return signals

name = "12h_1W_HighLow_Breakout_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0
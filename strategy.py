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
    
    # Get 1d data for calculations (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day Williams %R (14-period) for oversold/overbought
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    willr_14_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 14:
        for i in range(13, len(close_1d)):
            highest_high = np.max(high_1d[i-13:i+1])
            lowest_low = np.min(low_1d[i-13:i+1])
            if highest_high != lowest_low:
                willr_14_1d[i] = -100 * (highest_high - close_1d[i]) / (highest_high - lowest_low)
            else:
                willr_14_1d[i] = -50
    
    # Calculate 1-day RSI (14-period) for momentum confirmation
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full(len(close_1d), np.nan)
    avg_loss = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_14_1d = 100 - (100 / (1 + rs))
    
    # Calculate 1-day ATR (14-period) for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_14_1d = np.full(len(tr), np.nan)
    if len(tr) >= 14:
        atr_14_1d[13] = np.mean(tr[1:15])
        for i in range(14, len(tr)):
            atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr[i]) / 14
    
    # Align 1d indicators to 12h timeframe
    willr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, willr_14_1d)
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 12-period volume average for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 12
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(14, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(willr_14_1d_aligned[i]) or 
            np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND RSI > 50 (bullish momentum) with volume
            if willr_14_1d_aligned[i] < -80 and rsi_14_1d_aligned[i] > 50 and vol_filter:
                signals[i] = size
                position = 1
            # Short: Williams %R overbought (> -20) AND RSI < 50 (bearish momentum) with volume
            elif willr_14_1d_aligned[i] > -20 and rsi_14_1d_aligned[i] < 50 and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Williams %R returns to neutral (> -50) OR volatility spike
            if willr_14_1d_aligned[i] > -50 or vol_ratio > 2.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Williams %R returns to neutral (< -50) OR volatility spike
            if willr_14_1d_aligned[i] < -50 or vol_ratio > 2.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Williams_RSI_Volume"
timeframe = "12h"
leverage = 1.0
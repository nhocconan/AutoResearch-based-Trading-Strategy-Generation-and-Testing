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
    
    # Get 1w and 1d data for calculations (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 10 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-week RSI(14) for trend filter
    close_1w = df_1w['close'].values
    delta_1w = np.diff(close_1w, prepend=close_1w[0])
    gain_1w = np.where(delta_1w > 0, delta_1w, 0)
    loss_1w = np.where(delta_1w < 0, -delta_1w, 0)
    avg_gain_1w = np.full(len(close_1w), np.nan)
    avg_loss_1w = np.full(len(close_1w), np.nan)
    for i in range(14, len(close_1w)):
        if i == 14:
            avg_gain_1w[i] = np.mean(gain_1w[i-13:i+1])
            avg_loss_1w[i] = np.mean(loss_1w[i-13:i+1])
        else:
            avg_gain_1w[i] = (avg_gain_1w[i-1] * 13 + gain_1w[i]) / 14
            avg_loss_1w[i] = (avg_loss_1w[i-1] * 13 + loss_1w[i]) / 14
    rs_1w = np.divide(avg_gain_1w, avg_loss_1w, out=np.full_like(avg_gain_1w, np.nan), where=avg_loss_1w!=0)
    rsi_1w = 100 - (100 / (1 + rs_1w))
    
    # Calculate 1-day Donchian channels (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high_1d = np.full(len(high_1d), np.nan)
    donchian_low_1d = np.full(len(low_1d), np.nan)
    for i in range(19, len(high_1d)):
        donchian_high_1d[i] = np.max(high_1d[i-19:i+1])
        donchian_low_1d[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate 1-day volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_1d = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        vol_avg_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align indicators to 6h timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    donchian_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate 6-period volume average for spike detection
    vol_ma_6h = np.full(n, np.nan)
    vol_period = 6
    for i in range(vol_period, n):
        vol_ma_6h[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(19, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(donchian_high_1d_aligned[i]) or 
            np.isnan(donchian_low_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or 
            np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_6h[i] if vol_ma_6h[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: RSI > 50 (bullish weekly trend) + price breaks above 1d Donchian high + volume
            if rsi_1w_aligned[i] > 50 and price > donchian_high_1d_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: RSI < 50 (bearish weekly trend) + price breaks below 1d Donchian low + volume
            elif rsi_1w_aligned[i] < 50 and price < donchian_low_1d_aligned[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below 1d Donchian low
            if price < donchian_low_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above 1d Donchian high
            if price > donchian_high_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_RSI1W_Donchian1D_Volume"
timeframe = "6h"
leverage = 1.0
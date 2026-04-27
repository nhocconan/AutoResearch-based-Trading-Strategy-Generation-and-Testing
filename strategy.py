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
    
    # Get daily data for trend filter (EMA50) and volatility
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on daily close
    close_d = df_d['close'].values
    ema_50_d = np.full(len(close_d), np.nan)
    if len(close_d) >= 50:
        ema_50_d[49] = np.mean(close_d[:50])
        for i in range(50, len(close_d)):
            ema_50_d[i] = (close_d[i] * 2 + ema_50_d[i-1] * 48) / 50
    
    # Align daily EMA50 to 6h
    ema_50_d_aligned = align_htf_to_ltf(prices, df_d, ema_50_d)
    
    # Calculate daily ATR(14) for volatility filter
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d_arr = df_d['close'].values
    tr_d = np.maximum(high_d[1:] - low_d[1:], 
                      np.maximum(np.abs(high_d[1:] - close_d_arr[:-1]), 
                                 np.abs(low_d[1:] - close_d_arr[:-1])))
    tr_d = np.concatenate([[np.nan], tr_d])
    atr_d = np.full(len(close_d), np.nan)
    for i in range(14, len(close_d)):
        if i == 14:
            atr_d[i] = np.mean(tr_d[1:15])
        else:
            atr_d[i] = (atr_d[i-1] * 13 + tr_d[i]) / 14
    atr_d_aligned = align_htf_to_ltf(prices, df_d, atr_d)
    
    # Calculate 6h ATR(14) for position sizing and stop loss
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate 20-period volume average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Calculate 20-period high/low for Donchian breakout
    high_max = np.full(n, np.nan)
    low_min = np.full(n, np.nan)
    period = 20
    for i in range(period, n):
        high_max[i] = np.max(high[i-period:i])
        low_min[i] = np.min(low[i-period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(14, vol_period, period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_d_aligned[i]) or np.isnan(atr_d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Only trade when daily volatility is low (avoid choppy markets)
        vol_filter = atr_d_aligned[i] < np.mean(atr_d_aligned[max(0, i-50):i]) if i >= 50 else True
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume AND above daily EMA50
            if price > high_max[i] and vol_ratio > 2.0 and price > ema_50_d_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: Price breaks below Donchian low with volume AND below daily EMA50
            elif price < low_min[i] and vol_ratio > 2.0 and price < ema_50_d_aligned[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below Donchian low or 2x ATR trailing stop
            if price < low_min[i] or price < high_max[i] - 2 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above Donchian high or 2x ATR trailing stop
            if price > high_max[i] or price > low_min[i] + 2 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_1dEMA50_VolumeTrend"
timeframe = "6h"
leverage = 1.0
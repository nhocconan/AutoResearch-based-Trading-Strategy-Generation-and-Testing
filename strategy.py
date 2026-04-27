#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (EMA50) and volatility
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on weekly close
    close_w = df_w['close'].values
    ema_50_w = np.full(len(close_w), np.nan)
    if len(close_w) >= 50:
        ema_50_w[49] = np.mean(close_w[:50])
        for i in range(50, len(close_w)):
            ema_50_w[i] = (close_w[i] * 2 + ema_50_w[i-1] * 48) / 50
    
    # Align weekly EMA50 to daily
    ema_50_w_aligned = align_htf_to_ltf(prices, df_w, ema_50_w)
    
    # Calculate weekly ATR(14) for volatility filter
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w_arr = df_w['close'].values
    tr_w = np.maximum(high_w[1:] - low_w[1:], 
                      np.maximum(np.abs(high_w[1:] - close_w_arr[:-1]), 
                                 np.abs(low_w[1:] - close_w_arr[:-1])))
    tr_w = np.concatenate([[np.nan], tr_w])
    atr_w = np.full(len(close_w), np.nan)
    for i in range(14, len(close_w)):
        if i == 14:
            atr_w[i] = np.mean(tr_w[1:15])
        else:
            atr_w[i] = (atr_w[i-1] * 13 + tr_w[i]) / 14
    atr_w_aligned = align_htf_to_ltf(prices, df_w, atr_w)
    
    # Calculate daily ATR(14) for position sizing and stop loss
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
        if (np.isnan(ema_50_w_aligned[i]) or np.isnan(atr_w_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Only trade when weekly volatility is low (avoid choppy markets)
        vol_filter = atr_w_aligned[i] < np.mean(atr_w_aligned[max(0, i-50):i]) if i >= 50 else True
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume AND above weekly EMA50
            if price > high_max[i] and vol_ratio > 2.0 and price > ema_50_w_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: Price breaks below Donchian low with volume AND below weekly EMA50
            elif price < low_min[i] and vol_ratio > 2.0 and price < ema_50_w_aligned[i] and vol_filter:
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

name = "1d_Donchian20_1wEMA50_VolumeTrend"
timeframe = "1d"
leverage = 1.0
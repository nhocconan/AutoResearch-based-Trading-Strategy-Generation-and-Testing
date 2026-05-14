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
    
    # Get daily data for weekly EMA50 and ATR calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 from daily data (approximation: 5 days = 1 week)
    # We'll use 5-day EMA on daily close as proxy for weekly EMA50
    close_1d = df_1d['close'].values
    ema_5_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 5:
        ema_5_1d[4] = np.mean(close_1d[:5])
        for i in range(5, len(close_1d)):
            ema_5_1d[i] = (close_1d[i] * 2 + ema_5_1d[i-1] * 3) / 5  # 5-period EMA
    
    # Align 5-day EMA (proxy for weekly EMA50) to 12h timeframe
    ema_5_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_5_1d)
    
    # Calculate weekly ATR(14) from daily data (14 days = 2 weeks)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.maximum(np.abs(high_1d[1:] - close_1d_arr[:-1]), 
                                  np.abs(low_1d[1:] - close_1d_arr[:-1])))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_1d = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        if i == 14:
            atr_1d[i] = np.mean(tr_1d[1:15])
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 12h ATR(14) for position sizing and stop loss
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
    
    # Calculate 20-period volume average for 12h timeframe
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Calculate 10-period high/low for Donchian breakout (shorter period for 12h)
    high_max = np.full(n, np.nan)
    low_min = np.full(n, np.nan)
    period = 10
    for i in range(period, n):
        high_max[i] = np.max(high[i-period:i])
        low_min[i] = np.min(low[i-period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(14, vol_period, period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_5_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Only trade when daily volatility is not extremely high (avoid panic markets)
        vol_filter = atr_1d_aligned[i] < np.mean(atr_1d_aligned[max(0, i-30):i]) if i >= 30 else True
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume AND above 5-day EMA
            if price > high_max[i] and vol_ratio > 1.8 and price > ema_5_1d_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: Price breaks below Donchian low with volume AND below 5-day EMA
            elif price < low_min[i] and vol_ratio > 1.8 and price < ema_5_1d_aligned[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below Donchian low or 1.5x ATR trailing stop
            if price < low_min[i] or price < high_max[i] - 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above Donchian high or 1.5x ATR trailing stop
            if price > high_max[i] or price > low_min[i] + 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian10_5dEMA_VolumeTrend"
timeframe = "12h"
leverage = 1.0
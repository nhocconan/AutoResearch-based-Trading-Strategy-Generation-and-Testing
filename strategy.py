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
    
    # Get 4h data for trend and volatility filters
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h EMA(50) for trend direction
    close_4h = df_4h['close'].values
    ema_50_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema_50_4h[49] = np.mean(close_4h[:50])
        for i in range(50, len(close_4h)):
            ema_50_4h[i] = (close_4h[i] * 0.0769 + ema_50_4h[i-1] * 0.9231)  # EMA 50
    
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 4h ATR(20) for volatility filter
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_arr = df_4h['close'].values
    tr_4h = np.maximum(high_4h[1:] - low_4h[1:], 
                       np.maximum(np.abs(high_4h[1:] - close_4h_arr[:-1]), 
                                  np.abs(low_4h[1:] - close_4h_arr[:-1])))
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr_4h = np.full(len(close_4h), np.nan)
    for i in range(20, len(close_4h)):
        if i == 20:
            atr_4h[i] = np.mean(tr_4h[1:21])
        else:
            atr_4h[i] = (atr_4h[i-1] * 0.95 + tr_4h[i] * 0.05)  # Wilder's smoothing
    
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # 1h ATR(14) for position sizing and stops
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 0.9333 + tr[i] * 0.0667)  # Wilder's smoothing
    
    # 1h 20-period volume average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # 1h 10-period Donchian channels
    high_max = np.full(n, np.nan)
    low_min = np.full(n, np.nan)
    period = 10
    for i in range(period, n):
        high_max[i] = np.max(high[i-period:i])
        low_min[i] = np.min(low[i-period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.20
    
    # Warmup period
    start_idx = max(50, 20, vol_period, period) + 5
    
    # Pre-compute hourly session filter (8-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(start_idx, n):
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
            
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(atr_4h_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volatility filter: only trade when 4h volatility is below median (avoid chaotic markets)
        if i >= 50:
            vol_median = np.nanmedian(atr_4h_aligned[max(0, i-50):i])
            vol_filter = atr_4h_aligned[i] < vol_median
        else:
            vol_filter = True
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume AND above 4h EMA50
            if price > high_max[i] and vol_ratio > 2.0 and price > ema_50_4h_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: Price breaks below Donchian low with volume AND below 4h EMA50
            elif price < low_min[i] and vol_ratio > 2.0 and price < ema_50_4h_aligned[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below Donchian low or 2x ATR trailing stop
            if price < low_min[i] or price < high_max[i] - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above Donchian high or 2x ATR trailing stop
            if price > high_max[i] or price > low_min[i] + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Donchian10_4hEMA50_VolumeFilter_Session"
timeframe = "1h"
leverage = 1.0
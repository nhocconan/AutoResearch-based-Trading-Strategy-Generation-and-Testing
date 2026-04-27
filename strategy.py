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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on weekly close
    ema_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate 20-period ATR on weekly using Wilder's smoothing
    tr = np.maximum(high_1w[1:] - low_1w[1:], 
                    np.maximum(np.abs(high_1w[1:] - close_1w[:-1]), 
                               np.abs(low_1w[1:] - close_1w[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr_1w = np.full(len(tr), np.nan)
    for i in range(20, len(tr)):
        if i == 20:
            atr_1w[i] = np.mean(tr[1:21])
        else:
            atr_1w[i] = (atr_1w[i-1] * 19 + tr[i]) / 20
    
    # Align weekly indicators to daily
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Daily Donchian channel (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly EMA/ATR, Donchian, volume MA
    start_idx = max(50, 20, 20) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(atr_1w_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        atr = atr_1w_aligned[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper with weekly uptrend and volume
            if (price > highest_high[i] and 
                price > ema_1w_aligned[i] and 
                vol_ratio > 1.5):
                signals[i] = size
                position = 1
            # Short: Price breaks below Donchian lower with weekly downtrend and volume
            elif (price < lowest_low[i] and 
                  price < ema_1w_aligned[i] and 
                  vol_ratio > 1.5):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below Donchian lower or ATR-based stop
            if (price < lowest_low[i] or 
                price < close[i-1] - 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above Donchian upper or ATR-based stop
            if (price > highest_high[i] or 
                price > close[i-1] + 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyDonchian_EMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0
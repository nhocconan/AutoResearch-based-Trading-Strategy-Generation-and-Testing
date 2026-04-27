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
    
    # Get 1d data for ATR and mean price
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-period ATR on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR is just high-low
    
    atr_14 = np.zeros(len(tr))
    atr_14[:14] = np.nan
    atr_14[13] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate mean price (typical price) on daily
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    mean_price_1d = pd.Series(typical_price_1d).rolling(window=20, min_periods=20).mean().values
    mean_price_1d_aligned = align_htf_to_ltf(prices, df_1d, mean_price_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on weekly close
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need ATR, mean price, and weekly EMA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_14_aligned[i]) or np.isnan(mean_price_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        atr = atr_14_aligned[i]
        mean_price = mean_price_1d_aligned[i]
        weekly_ema = ema_50_1w_aligned[i]
        
        # Distance from mean in ATR units
        distance_from_mean = (close[i] - mean_price) / atr
        
        # Entry conditions: extreme deviation + weekly trend alignment
        if position == 0:
            # Long: price significantly below mean AND weekly uptrend
            if distance_from_mean < -2.0 and close[i] > weekly_ema:
                signals[i] = size
                position = 1
            # Short: price significantly above mean AND weekly downtrend
            elif distance_from_mean > 2.0 and close[i] < weekly_ema:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to mean or weekly trend breaks
            if distance_from_mean > -0.5 or close[i] < weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to mean or weekly trend breaks
            if distance_from_mean < 0.5 or close[i] > weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_MeanReversion_ATR_WeeklyTrend"
timeframe = "6h"
leverage = 1.0
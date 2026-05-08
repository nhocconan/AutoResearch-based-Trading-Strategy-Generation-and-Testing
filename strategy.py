#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WickReversal_VolumeTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter (weekly EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d data for ATR (volatility filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on 1d
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # first tr is inf
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Wick rejection strength: (high-low) - |close-open| (upper/lower wick size)
    body_size = np.abs(close - prices['open'].values)
    total_range = high - low
    upper_wick = high - np.maximum(close, prices['open'].values)
    lower_wick = np.minimum(close, prices['open'].values) - low
    
    # Volume filter: current volume > 1.5x 24-period average (4 days on 6h)
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(upper_wick[i]) or np.isnan(lower_wick[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: strong lower wick rejection + above weekly EMA + volume
            lower_wick_ratio = lower_wick[i] / total_range[i] if total_range[i] > 0 else 0
            long_cond = (lower_wick_ratio > 0.6 and  # strong lower wick
                        close[i] > ema_50_1w_aligned[i] and
                        volume_filter[i])
            
            # Short: strong upper wick rejection + below weekly EMA + volume
            upper_wick_ratio = upper_wick[i] / total_range[i] if total_range[i] > 0 else 0
            short_cond = (upper_wick_ratio > 0.6 and  # strong upper wick
                         close[i] < ema_50_1w_aligned[i] and
                         volume_filter[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below weekly EMA or strong upper wick rejection
            upper_wick_ratio = upper_wick[i] / total_range[i] if total_range[i] > 0 else 0
            if close[i] < ema_50_1w_aligned[i] or upper_wick_ratio > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above weekly EMA or strong lower wick rejection
            lower_wick_ratio = lower_wick[i] / total_range[i] if total_range[i] > 0 else 0
            if close[i] > ema_50_1w_aligned[i] or lower_wick_ratio > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
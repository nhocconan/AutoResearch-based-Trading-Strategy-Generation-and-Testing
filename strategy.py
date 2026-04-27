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
    
    # Get daily data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on daily data
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(len(tr)):
        if i < 13:
            atr_1d[i] = np.mean(tr[:i+1]) if i > 0 else tr[i]
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate ATR-based volatility filter: current ATR > 1.5 * average ATR(20)
    atr_ma_20 = np.full(len(df_1d), np.nan)
    for i in range(20, len(atr_1d)):
        atr_ma_20[i] = np.mean(atr_1d[i-20:i])
    
    vol_filter = np.full(len(df_1d), False)
    for i in range(20, len(df_1d)):
        if not np.isnan(atr_1d[i]) and not np.isnan(atr_ma_20[i]) and atr_ma_20[i] > 0:
            vol_filter[i] = atr_1d[i] > 1.5 * atr_ma_20[i]
    
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter)
    
    # Calculate daily EMA(50) for trend filter
    ema_50 = np.full(len(df_1d), np.nan)
    alpha = 2 / (50 + 1)
    for i in range(len(close_1d)):
        if i < 49:
            ema_50[i] = np.mean(close_1d[:i+1]) if i > 0 else close_1d[i]
        else:
            if np.isnan(ema_50[i-1]):
                ema_50[i] = np.mean(close_1d[i-49:i+1])
            else:
                ema_50[i] = close_1d[i] * alpha + ema_50[i-1] * (1 - alpha)
    
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Bollinger Bands (20, 2) on daily data
    bb_mid = np.full(len(df_1d), np.nan)
    bb_std = np.full(len(df_1d), np.nan)
    for i in range(19, len(close_1d)):
        bb_mid[i] = np.mean(close_1d[i-19:i+1])
        bb_std[i] = np.std(close_1d[i-19:i+1])
    
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(bb_upper_aligned[i]) or
            np.isnan(bb_lower_aligned[i]) or
            np.isnan(vol_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above upper BB with volatility expansion and price above EMA50
            if (vol_filter_aligned[i] and 
                price > bb_upper_aligned[i] and 
                close[i-1] <= bb_upper_aligned[i-1] and  # just broke above
                price > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower BB with volatility expansion and price below EMA50
            elif (vol_filter_aligned[i] and 
                  price < bb_lower_aligned[i] and 
                  close[i-1] >= bb_lower_aligned[i-1] and  # just broke below
                  price < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price closes below EMA50 or volatility contraction
            if (price < ema_50_aligned[i] or 
                not vol_filter_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price closes above EMA50 or volatility contraction
            if (price > ema_50_aligned[i] or 
                not vol_filter_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "12h_BollingerBreakout_VolatilityFilter_EMA50_Trend_v1"
timeframe = "12h"
leverage = 1.0
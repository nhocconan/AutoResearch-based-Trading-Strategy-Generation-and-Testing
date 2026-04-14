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
    
    # Load daily data once for trend filter and pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily EMA(20) for trend
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_20_1d = close_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = high_1d[1:] - low_1d[:-1]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Daily pivot points (standard calculation)
    # Pivot = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    h_minus_l_1d = high_1d - low_1d
    r2_1d = pivot_1d + h_minus_l_1d
    s2_1d = pivot_1d - h_minus_l_1d
    r3_1d = high_1d + 2 * (pivot_1d - low_1d)
    s3_1d = low_1d - 2 * (high_1d - pivot_1d)
    
    # Create arrays for alignment
    ema_20_1d_arr = ema_20_1d
    atr_1d_arr = atr_1d
    pivot_1d_arr = pivot_1d
    r1_1d_arr = r1_1d
    s1_1d_arr = s1_1d
    r2_1d_arr = r2_1d
    s2_1d_arr = s2_1d
    r3_1d_arr = r3_1d
    s3_1d_arr = s3_1d
    
    # Calculate median volume for volume spike filter
    vol_median = np.nanmedian(volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # Get aligned daily data
        ema_20_1d_i = align_htf_to_ltf(prices, df_1d, ema_20_1d_arr)[i]
        atr_1d_i = align_htf_to_ltf(prices, df_1d, atr_1d_arr)[i]
        pivot_1d_i = align_htf_to_ltf(prices, df_1d, pivot_1d_arr)[i]
        r1_1d_i = align_htf_to_ltf(prices, df_1d, r1_1d_arr)[i]
        s1_1d_i = align_htf_to_ltf(prices, df_1d, s1_1d_arr)[i]
        r2_1d_i = align_htf_to_ltf(prices, df_1d, r2_1d_arr)[i]
        s2_1d_i = align_htf_to_ltf(prices, df_1d, s2_1d_arr)[i]
        r3_1d_i = align_htf_to_ltf(prices, df_1d, r3_1d_arr)[i]
        s3_1d_i = align_htf_to_ltf(prices, df_1d, s3_1d_arr)[i]
        
        if np.isnan(ema_20_1d_i) or np.isnan(atr_1d_i) or \
           np.isnan(pivot_1d_i) or np.isnan(r1_1d_i) or np.isnan(s1_1d_i) or \
           np.isnan(r2_1d_i) or np.isnan(s2_1d_i) or np.isnan(r3_1d_i) or np.isnan(s3_1d_i):
            continue
        
        # Volatility filter: only trade when daily ATR is above median (avoid choppy markets)
        atr_median = np.nanmedian(atr_1d_arr)
        vol_filter = atr_1d_i > 0.8 * atr_median
        
        # Volume spike filter
        volume_spike = volume[i] > 1.5 * vol_median
        
        # Long conditions:
        # 1. Price above daily EMA20 (uptrend)
        # 2. Price breaks above R1 with volume (breakout)
        # 3. Not overextended (below R2 to avoid chasing)
        if position == 0 and vol_filter and volume_spike:
            if close[i] > ema_20_1d_i and close[i] > r1_1d_i and close[i] < r2_1d_i:
                position = 1
                signals[i] = position_size
            # Short conditions:
            # 1. Price below daily EMA20 (downtrend)
            # 2. Price breaks below S1 with volume (breakdown)
            # 3. Not overextended (above S2 to avoid chasing)
            elif close[i] < ema_20_1d_i and close[i] < s1_1d_i and close[i] > s2_1d_i:
                position = -1
                signals[i] = -position_size
        
        # Exit conditions:
        # Long exit: price crosses below pivot or hits S1 (mean reversion to mean)
        elif position == 1:
            if close[i] < pivot_1d_i or close[i] < s1_1d_i:
                position = 0
                signals[i] = 0.0
        # Short exit: price crosses above pivot or hits R1 (mean reversion to mean)
        elif position == -1:
            if close[i] > pivot_1d_i or close[i] > r1_1d_i:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_DailyPivotBreakout_EMA20_VolumeFilter"
timeframe = "6h"
leverage = 1.0
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
    
    # Load weekly data once for trend filter and pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly EMA(20) for trend
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Weekly ATR(14) for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    tr1 = high_1w[1:] - low_1w[:-1]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Weekly pivot points (standard calculation)
    # Pivot = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    h_minus_l_1w = high_1w - low_1w
    r2_1w = pivot_1w + h_minus_l_1w
    s2_1w = pivot_1w - h_minus_l_1w
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    
    # Create arrays for alignment
    ema_20_1w_arr = ema_20_1w
    atr_1w_arr = atr_1w
    pivot_1w_arr = pivot_1w
    r1_1w_arr = r1_1w
    s1_1w_arr = s1_1w
    r2_1w_arr = r2_1w
    s2_1w_arr = s2_1w
    r3_1w_arr = r3_1w
    s3_1w_arr = s3_1w
    
    # Calculate median volume for volume spike filter
    vol_median = np.nanmedian(volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # Get aligned weekly data
        ema_20_1w_i = align_htf_to_ltf(prices, df_1w, ema_20_1w_arr)[i]
        atr_1w_i = align_htf_to_ltf(prices, df_1w, atr_1w_arr)[i]
        pivot_1w_i = align_htf_to_ltf(prices, df_1w, pivot_1w_arr)[i]
        r1_1w_i = align_htf_to_ltf(prices, df_1w, r1_1w_arr)[i]
        s1_1w_i = align_htf_to_ltf(prices, df_1w, s1_1w_arr)[i]
        r2_1w_i = align_htf_to_ltf(prices, df_1w, r2_1w_arr)[i]
        s2_1w_i = align_htf_to_ltf(prices, df_1w, s2_1w_arr)[i]
        r3_1w_i = align_htf_to_ltf(prices, df_1w, r3_1w_arr)[i]
        s3_1w_i = align_htf_to_ltf(prices, df_1w, s3_1w_arr)[i]
        
        if np.isnan(ema_20_1w_i) or np.isnan(atr_1w_i) or \
           np.isnan(pivot_1w_i) or np.isnan(r1_1w_i) or np.isnan(s1_1w_i) or \
           np.isnan(r2_1w_i) or np.isnan(s2_1w_i) or np.isnan(r3_1w_i) or np.isnan(s3_1w_i):
            continue
        
        # Volatility filter: only trade when weekly ATR is above median (avoid choppy markets)
        atr_median = np.nanmedian(atr_1w_arr)
        vol_filter = atr_1w_i > 0.8 * atr_median
        
        # Volume spike filter
        volume_spike = volume[i] > 1.5 * vol_median
        
        # Long conditions:
        # 1. Price above weekly EMA20 (uptrend)
        # 2. Price breaks above R1 with volume (breakout)
        # 3. Not overextended (below R2 to avoid chasing)
        if position == 0 and vol_filter and volume_spike:
            if close[i] > ema_20_1w_i and close[i] > r1_1w_i and close[i] < r2_1w_i:
                position = 1
                signals[i] = position_size
            # Short conditions:
            # 1. Price below weekly EMA20 (downtrend)
            # 2. Price breaks below S1 with volume (breakdown)
            # 3. Not overextended (above S2 to avoid chasing)
            elif close[i] < ema_20_1w_i and close[i] < s1_1w_i and close[i] > s2_1w_i:
                position = -1
                signals[i] = -position_size
        
        # Exit conditions:
        # Long exit: price crosses below pivot or hits S1 (mean reversion to mean)
        elif position == 1:
            if close[i] < pivot_1w_i or close[i] < s1_1w_i:
                position = 0
                signals[i] = 0.0
        # Short exit: price crosses above pivot or hits R1 (mean reversion to mean)
        elif position == -1:
            if close[i] > pivot_1w_i or close[i] > r1_1w_i:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivotBreakout_EMA20_VolumeFilter"
timeframe = "6h"
leverage = 1.0
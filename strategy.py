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
    
    # Get daily data for pivot points and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily pivot points (Classic formula)
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    # R2 = P + (H - L), S2 = P - (H - L)
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    r3_1d = high_1d + 2 * (pivot_1d - low_1d)
    s3_1d = low_1d - 2 * (high_1d - pivot_1d)
    
    # Align pivot levels to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Volume filter: daily volume > 20-period average (volume spike)
    vol_ma_20 = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_20[i] = np.mean(volume_1d[i-20:i])
    
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    volume_spike = np.full(n, False)
    valid_vol = (~np.isnan(volume_1d)) & (~np.isnan(vol_ma_20_aligned))
    volume_spike[valid_vol] = volume_1d[valid_vol] > (vol_ma_20_aligned[valid_vol] * 1.5)
    
    # Calculate 6-period RSI for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(6, n):
        if i == 6:
            avg_gain[i] = np.mean(gain[1:7])
            avg_loss[i] = np.mean(loss[1:7])
        else:
            avg_gain[i] = (avg_gain[i-1] * 5 + gain[i]) / 6
            avg_loss[i] = (avg_loss[i-1] * 5 + loss[i]) / 6
    
    rs = np.full(n, np.nan)
    valid_rsi = (~np.isnan(avg_gain)) & (~np.isnan(avg_loss)) & (avg_loss > 0)
    rs[valid_rsi] = avg_gain[valid_rsi] / avg_loss[valid_rsi]
    rsi_6 = np.full(n, np.nan)
    rsi_6[valid_rsi] = 100 - (100 / (1 + rs[valid_rsi]))
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need enough data for RSI and volume MA
    start_idx = max(6, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_6[i]) or 
            np.isnan(pivot_1d_aligned[i]) or
            np.isnan(r1_1d_aligned[i]) or
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(r2_1d_aligned[i]) or
            np.isnan(s2_1d_aligned[i]) or
            np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long setup: price near S1/S2 with RSI oversold and volume spike
            near_support = (price <= s1_1d_aligned[i] * 1.02 and price >= s2_1d_aligned[i] * 0.98) or \
                          (price <= s2_1d_aligned[i] * 1.02 and price >= s3_1d_aligned[i] * 0.98)
            rsi_oversold = rsi_6[i] < 30
            
            if near_support and rsi_oversold and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short setup: price near R1/R2 with RSI overbought and volume spike
            elif (price >= r1_1d_aligned[i] * 0.98 and price <= r2_1d_aligned[i] * 1.02) or \
                 (price >= r2_1d_aligned[i] * 0.98 and price <= r3_1d_aligned[i] * 1.02):
                rsi_overbought = rsi_6[i] > 70
                if rsi_overbought and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price reaches pivot or RSI overbought
            if price >= pivot_1d_aligned[i] * 0.99 or rsi_6[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches pivot or RSI oversold
            if price <= pivot_1d_aligned[i] * 1.01 or rsi_6[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_PivotMeanReversion_RSI6_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0
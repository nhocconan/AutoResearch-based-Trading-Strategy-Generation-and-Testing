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
    
    # Get 1d HTF data once before loop for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily pivot points (standard floor trader's pivots)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1 = 2 * pp - low_1d
    s1 = 2 * pp - high_1d
    # R2 = P + (H - L), S2 = P - (H - L)
    r2 = pp + (high_1d - low_1d)
    s2 = pp - (high_1d - low_1d)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    r3 = high_1d + 2 * (pp - low_1d)
    s3 = low_1d - 2 * (high_1d - pp)
    
    # Align 1d pivot levels to 6h
    pp_6h = align_htf_to_ltf(prices, df_1d, pp)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Get 1w HTF data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 6h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Pre-compute session filter (00-23 UTC - 6h bars less session sensitive)
    hours = prices.index.hour
    in_session = (hours >= 0) & (hours <= 23)  # Always true for 6h, but keeps structure
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(r3_6h[i]) or
            np.isnan(s3_6h[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 6h price breaks above R1 with volume
        # 2. 1w EMA(50) trend filter: price above EMA50 (bullish bias)
        # 3. Volume confirmation: volume > 1.3x average
        # 4. Not already above R2 (avoid chasing)
        if (close[i] > r1_6h[i] and
            close[i] <= r2_6h[i] and  # Breakout zone between R1 and R2
            close[i] > ema_50_1w_aligned[i] and
            volume_ratio[i] > 1.3):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 6h price breaks below S1 with volume
        # 2. 1w EMA(50) trend filter: price below EMA50 (bearish bias)
        # 3. Volume confirmation: volume > 1.3x average
        # 4. Not already below S2 (avoid chasing)
        elif (close[i] < s1_6h[i] and
              close[i] >= s2_6h[i] and  # Breakdown zone between S1 and S2
              close[i] < ema_50_1w_aligned[i] and
              volume_ratio[i] > 1.3):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_1d_Pivot_R1S1_Breakout_1w_EMA50_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0
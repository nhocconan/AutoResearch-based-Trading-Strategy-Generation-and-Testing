#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data ONCE
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1-day pivot levels (standard formula)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    r3 = high_1d + 2 * (pivot - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 6-day EMA for trend filter on 1d closes
    close_1d_series = pd.Series(close_1d)
    ema_6_1d = close_1d_series.ewm(span=6, adjust=False, min_periods=6).mean().values
    ema_6_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_6_1d)
    
    # Calculate 6h ATR(14) for volatility filter
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 6h EMA(20) for trend confirmation
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(ema_6_1d_aligned[i]) or np.isnan(atr_14[i]) or
            np.isnan(ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_6_1d_val = ema_6_1d_aligned[i]
        atr_val = atr_14[i]
        ema_20_val = ema_20[i]
        
        # Trend filter: 1d EMA6 slope (simplified as price vs EMA6)
        uptrend = price > ema_6_1d_val
        downtrend = price < ema_6_1d_val
        
        if position == 0:
            # Long: price breaks above R1 with uptrend + volatility filter
            if price > r1_val and uptrend and atr_val < np.nanpercentile(atr_14[max(0, i-49):i+1], 70):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with downtrend + volatility filter
            elif price < s1_val and downtrend and atr_val < np.nanpercentile(atr_14[max(0, i-49):i+1], 70):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below S1 or volatility spike
            if price < s1_val or atr_val > np.nanpercentile(atr_14[max(0, i-49):i+1], 80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above R1 or volatility spike
            if price > r1_val or atr_val > np.nanpercentile(atr_14[max(0, i-49):i+1], 80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_Pivot_R1S1_Breakout_TrendFilter"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA to 6h timeframe
    ema_50_1w_6h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point calculation (standard)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = pivot + (high_1d - low_1d)
    s1 = pivot - (high_1d - low_1d)
    r2 = pivot + 2 * (high_1d - low_1d)
    s2 = pivot - 2 * (high_1d - low_1d)
    r3 = pivot + 3 * (high_1d - low_1d)
    s3 = pivot - 3 * (high_1d - low_1d)
    r4 = pivot + 4 * (high_1d - low_1d)
    s4 = pivot - 4 * (high_1d - low_1d)
    
    # Align pivots to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # 6h ATR(14) for volatility filter
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter (20-period MA)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 1.5 * vol_ma20  # Moderate volume surge
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_6h[i]) or np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(r2_6h[i]) or
            np.isnan(r3_6h[i]) or np.isnan(r4_6h[i]) or np.isnan(s1_6h[i]) or np.isnan(s2_6h[i]) or
            np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(atr[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above S4 with volume surge, above weekly EMA50
            if (close[i] > s4_6h[i] and vol_surge[i] and close[i] > ema_50_1w_6h[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below R4 with volume surge, below weekly EMA50
            elif (close[i] < r4_6h[i] and vol_surge[i] and close[i] < ema_50_1w_6h[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses opposite pivot level or volatility drops significantly
            if position == 1:
                if close[i] < pivot_6h[i] or atr[i] < 0.5 * atr[i-1]:  # Volatility drop filter
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > pivot_6h[i] or atr[i] < 0.5 * atr[i-1]:  # Volatility drop filter
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Pivot_S4_R4_Breakout_1wEMA50_Trend_VolumeSurge_v1"
timeframe = "6h"
leverage = 1.0
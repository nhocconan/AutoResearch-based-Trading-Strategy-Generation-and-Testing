#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Weekly data for pivot points (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Pivot Points (S1, R1)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = pivot_1w + (high_1w - low_1w)
    s1_1w = pivot_1w - (high_1w - low_1w)
    
    # Align weekly pivot levels to 4h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # 4h ATR for volatility filter (14-period)
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
    vol_surge = prices['volume'].values > 2.0 * vol_ma20
    
    # 4h EMA34 for trend filter
    ema34 = pd.Series(close).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma20[i]) or np.isnan(ema34[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above S1 with volume surge and above EMA34
            if close[i] > s1_1w_aligned[i] and vol_surge[i] and close[i] > ema34[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below R1 with volume surge and below EMA34
            elif close[i] < r1_1w_aligned[i] and vol_surge[i] and close[i] < ema34[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to pivot level or volatility drops significantly
            if position == 1:
                if close[i] < pivot_1w_aligned[i] or atr[i] < 0.3 * atr[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > pivot_1w_aligned[i] or atr[i] < 0.3 * atr[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Pivot_S1_R1_Breakout_1w_Pivot_VolumeTrend_v1"
timeframe = "4h"
leverage = 1.0
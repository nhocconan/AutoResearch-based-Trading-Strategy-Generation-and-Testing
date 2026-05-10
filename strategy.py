#!/usr/bin/env python3
# 1d_Weekly_Camarilla_R1_S1_Breakout_With_Volume
# Hypothesis: Weekly Camarilla R1/S1 breakouts with daily volume confirmation and ADX trend filter provide edge in trending and ranging markets. Designed for low trade frequency (target: 10-20/year) with discrete sizing to minimize fee drift.

name = "1d_Weekly_Camarilla_R1_S1_Breakout_With_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly Camarilla levels (R1, S1)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r1_1w = close_1w + (range_1w * 1.0833)
    s1_1w = close_1w - (range_1w * 1.0833)
    
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Daily volume filter (20-period SMA)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX(14) for trend strength
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 14 + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 14 + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient history
    
    for i in range(start_idx, n):
        if np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or \
           np.isnan(vol_sma_20[i]) or np.isnan(adx[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_ok = volume[i] > vol_sma_20[i]
        trend_strong = adx[i] > 20
        
        if position == 0:
            # Long: break above weekly R1 with volume and trend
            if close[i] > r1_1w_aligned[i] and volume_ok and trend_strong:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1 with volume and trend
            elif close[i] < s1_1w_aligned[i] and volume_ok and trend_strong:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below weekly S1 or trend weakens
            if close[i] < s1_1w_aligned[i] or adx[i] < 18:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above weekly R1 or trend weakens
            if close[i] > r1_1w_aligned[i] or adx[i] < 18:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
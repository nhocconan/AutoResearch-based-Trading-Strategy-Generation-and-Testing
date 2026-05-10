#!/usr/bin/env python3
"""
1D_KAMA_Direction_1wTrend_Filter_Volume
Hypothesis: KAMA (Kaufman Adaptive Moving Average) on 1d identifies trend direction; 1w EMA200 filters long/short bias; volume spike >2x average confirms momentum.
Works in bull markets (trend following) and bear markets (avoids counter-trend trades via 1w filter). Target: 15-25 trades/year.
"""

name = "1D_KAMA_Direction_1wTrend_Filter_Volume"
timeframe = "1d"
leverage = 1.0

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
    
    # --- 1d KAMA (trend) ---
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum |close[t] - close[t-1]| over 10
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # --- 1w EMA200 (trend filter) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    ema_200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # --- Volume filter ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10)  # warmup for volume and KAMA
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        price_above_ema200w = close[i] > ema_200_1w_aligned[i]
        price_below_ema200w = close[i] < ema_200_1w_aligned[i]
        
        if position == 0:
            # Long: price > KAMA, price > 1w EMA200, volume spike
            if price_above_kama and price_above_ema200w and volume[i] > vol_threshold[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA, price < 1w EMA200, volume spike
            elif price_below_kama and price_below_ema200w and volume[i] > vol_threshold[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < KAMA or volume drops
            if price_below_kama or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > KAMA or volume drops
            if price_above_kama or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
"""
1d_KAMA_AdaptiveTrend_12hVWAP_Volume
Hypothesis: KAMA adapts to market noise - slow in ranging markets (reducing whipsaws), fast in trends.
Combines with 12h VWAP for institutional sentiment and volume confirmation.
Works in bull (trend following) and bear (adaptive filtering reduces losses during chop).
Target: 20-40 trades/year to minimize fee drag.
"""

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
    
    # Get 1d data for KAMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (adaptive moving average) on 1d
    # Parameters: ER period=10, fast=2, slow=30
    er_period = 10
    fast_sc = 2
    slow_sc = 30
    
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, er_period))  # |close - close_er_period|
    volatility = np.zeros(len(close_1d))
    for i in range(er_period, len(close_1d)):
        volatility[i] = np.sum(np.abs(np.diff(close_1d[i-er_period+1:i+1])))
    
    er = np.zeros(len(close_1d))
    er[er_period:] = change / np.where(volatility[er_period:] != 0, volatility[er_period:], 1)
    
    # Smoothing constant
    sc = np.zeros(len(close_1d))
    sc[er_period:] = (er[er_period:] * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    
    # KAMA
    kama = np.full(len(close_1d), np.nan)
    if len(close_1d) > er_period:
        kama[er_period] = close_1d[er_period]
        for i in range(er_period + 1, len(close_1d)):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Get 12h data for VWAP
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    typical_price_12h = (df_12h['high'].values + df_12h['low'].values + df_12h['close'].values) / 3
    vwap_num = np.cumsum(typical_price_12h * df_12h['volume'].values)
    vwap_den = np.cumsum(df_12h['volume'].values)
    vwap_12h = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Align indicators to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    
    # Volume confirmation on 4h
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position
    
    start_idx = max(er_period + 1, 19)
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(vwap_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: price > KAMA AND price > VWAP with volume
            if price > kama_aligned[i] and price > vwap_12h_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: price < KAMA AND price < VWAP with volume
            elif price < kama_aligned[i] and price < vwap_12h_aligned[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price < KAMA OR price < VWAP
            if price < kama_aligned[i] or price < vwap_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price > KAMA OR price > VWAP
            if price > kama_aligned[i] or price > vwap_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_AdaptiveTrend_12hVWAP_Volume"
timeframe = "4h"
leverage = 1.0
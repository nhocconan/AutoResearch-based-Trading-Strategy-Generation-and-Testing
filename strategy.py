#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_Trend_Volume_1d"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA trend and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # KAMA (Kaufman Adaptive Moving Average) calculation
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(df_1d['close'], n=10))
    volatility = np.sum(np.abs(np.diff(df_1d['close'])), axis=1)
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = np.power(er * (2/2 - 2/30) + 2/30, 2)  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(df_1d['close'], np.nan, dtype=float)
    kama[29] = df_1d['close'].iloc[29]  # start at index 29
    for i in range(30, len(df_1d)):
        kama[i] = kama[i-1] + sc[i] * (df_1d['close'].iloc[i] - kama[i-1])
    
    # Volume filter: current volume > 1.3 * 20-day average
    vol_series = pd.Series(df_1d['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = df_1d['volume'].values > (vol_ma * 1.3)
    
    # Align to 4h
    kama_4h = align_htf_to_ltf(prices, df_1d, kama)
    volume_filter_4h = align_htf_to_ltf(prices, df_1d, volume_filter)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30  # Need enough data for KAMA
    
    for i in range(start_idx, n):
        if np.isnan(kama_4h[i]) or np.isnan(volume_filter_4h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = kama_4h[i]
        vol_filter = volume_filter_4h[i]
        
        if position == 0:
            # Enter long: price above KAMA with volume
            if close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA with volume
            elif close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below KAMA
            if close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA
            if close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
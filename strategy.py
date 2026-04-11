#!/usr/bin/env python3
"""
4h_12h_kama_trend_volume_v1
Strategy: 4h KAMA trend following with 12h volume confirmation
Timeframe: 4h
Leverage: 1.0
Hypothesis: Use KAMA (adaptive moving average) on 4h to capture trend changes; require 12h volume surge (>2x average) to confirm institutional interest; only trade in direction of 4h KAMA slope. Designed to work in bull markets (trend continuation) and bear markets (sharp reversals on volume) by combining adaptive trend with volume confirmation. Low-frequency design targets 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_kama_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # === 4h KAMA (Adaptive Moving Average) ===
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    # Pad arrays to match length
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(1, np.nan), volatility])
    volatility = np.concatenate([np.full(9, np.nan), volatility[10:]]) if len(volatility) > 10 else np.full_like(change, np.nan)
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    fastest_sc = 2 / (2 + 1)   # for fast EMA (2-period)
    slowest_sc = 2 / (30 + 1)  # for slow EMA (30-period)
    sc = (er * (fastest_sc - slowest_sc) + slowest_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # KAMA slope (trend direction)
    kama_slope = np.diff(kama, prepend=kama[0])
    
    # === 12h Volume Filter ===
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_12h_ratio = vol_12h / vol_ma_12h
    vol_12h_ratio_aligned = align_htf_to_ltf(prices, df_12h, vol_12h_ratio)
    
    # Volume confirmation: 12h volume > 2x average
    volume_surge = vol_12h_ratio_aligned > 2.0
    
    # Session filter: active trading hours (0-23 UTC covers all major sessions)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 0) & (hours <= 23)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(kama[i]) or np.isnan(kama_slope[i]) or
            np.isnan(volume_surge[i]) or not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long conditions: price above KAMA, KAMA sloping up, volume surge on 12h
        long_signal = (close[i] > kama[i]) and (kama_slope[i] > 0) and volume_surge[i]
        
        # Short conditions: price below KAMA, KAMA sloping down, volume surge on 12h
        short_signal = (close[i] < kama[i]) and (kama_slope[i] < 0) and volume_surge[i]
        
        # Exit when price crosses KAMA in opposite direction (trend change)
        exit_long = position == 1 and close[i] < kama[i]
        exit_short = position == -1 and close[i] > kama[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
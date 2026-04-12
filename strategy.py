#!/usr/bin/env python3
"""
4h_12h_RVI_Trend_Signal
Hypothesis: Use 4h Relative Vigor Index (RVI) with 12h trend filter (price > 12h EMA50) for trend-following entries. RVI > 0.05 triggers long, RVI < -0.05 triggers short. Volume confirmation (>1.5x 20-period average) filters weak moves. Designed for low-frequency, high-probability trades in both bull and bear markets. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_RVI_Trend_Signal"
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
    
    # === RVI CALCULATION ===
    # RVI = (close - open) / (high - low) smoothed
    numerator = close - prices['open'].values
    denominator = high - low
    # Avoid division by zero
    rvi_raw = np.where(denominator != 0, numerator / denominator, 0.0)
    
    # Smooth with EMA (period=10)
    rvi_series = pd.Series(rvi_raw)
    rvi_ema = rvi_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # === 12h EMA50 TREND FILTER ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === VOLUME FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(rvi_ema[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions
        long_signal = (rvi_ema[i] > 0.05) and (close[i] > ema_50_12h_aligned[i]) and (vol_ratio[i] > 1.5)
        short_signal = (rvi_ema[i] < -0.05) and (close[i] < ema_50_12h_aligned[i]) and (vol_ratio[i] > 1.5)
        
        # Exit: RVI crosses back toward zero
        exit_long = (position == 1) and (rvi_ema[i] < 0.0)
        exit_short = (position == -1) and (rvi_ema[i] > 0.0)
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
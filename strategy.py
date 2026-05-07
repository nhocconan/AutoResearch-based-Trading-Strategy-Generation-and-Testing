#!/usr/bin/env python3
"""
4h_EquiVolume_Trend_Filter_v1
Hypothesis: Uses EquiVolume-weighted moving average on 4h timeframe for trend direction,
combined with volume confirmation and 1-day trend filter to reduce false signals.
EquiVolume gives more weight to high-volume periods, making it effective in both
trending and ranging markets. Targets 20-40 trades/year to minimize fee drag.
"""

name = "4h_EquiVolume_Trend_Filter_v1"
timeframe = "4h"
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
    
    # Calculate EquiVolume-weighted moving average (EMA-like)
    # EquiVolume: price weighted by volume
    # Using 21-period for balance between responsiveness and smoothness
    pv = close * volume  # price-volume product
    vol_sum = pd.Series(volume).rolling(window=21, min_periods=21).sum().values
    pv_sum = pd.Series(pv).rolling(window=21, min_periods=21).sum().values
    eva = np.divide(pv_sum, vol_sum, out=np.zeros_like(pv_sum), where=vol_sum!=0)
    
    # 1-day trend filter: EMA of daily close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(21, n):
        # Skip if any critical value is NaN
        if (np.isnan(eva[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above EVA AND above 1-day EMA with volume confirmation
            if close[i] > eva[i] and close[i] > ema_1d_aligned[i] and volume[i] > vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below EVA AND below 1-day EMA with volume confirmation
            elif close[i] < eva[i] and close[i] < ema_1d_aligned[i] and volume[i] > vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below EVA OR below 1-day EMA
            if close[i] < eva[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above EVA OR above 1-day EMA
            if close[i] > eva[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
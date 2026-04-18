#!/usr/bin/env python3
"""
12h_KAMA_1dTrend_VolumeSpike
Hypothesis: On 12h timeframe, KAMA direction combined with 1d trend filter and volume spike
captures momentum in both bull and bear markets. KAMA adapts to market noise, reducing whipsaw.
1d trend filter ensures alignment with higher timeframe momentum. Volume spike confirms
institutional participation. Designed for low trade frequency (12-37/year) to minimize fee drag.
"""

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
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close']
    # KAMA on 1d for trend direction
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.where(volatility.sum() != 0, change / volatility.sum(), 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    kama_trend = kama_1d > np.roll(kama_1d, 1)  # rising KAMA = uptrend
    kama_trend = np.concatenate([[False], kama_trend[:-1]])  # avoid look-ahead
    kama_trend_aligned = align_htf_to_ltf(prices, df_1d, kama_trend.astype(float))
    
    # Volume spike: volume > 1.5 * 20-period MA on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Price above/below KAMA on 12h for entry
    change_12h = np.abs(np.diff(close, prepend=close[0]))
    volatility_12h = np.abs(np.diff(close))
    er_12h = np.where(volatility_12h.sum() != 0, change_12h / volatility_12h.sum(), 0)
    sc_12h = (er_12h * (2/2 - 2/30) + 2/30) ** 2
    kama_12h = np.zeros_like(close)
    kama_12h[0] = close[0]
    for i in range(1, len(close)):
        kama_12h[i] = kama_12h[i-1] + sc_12h[i] * (close[i] - kama_12h[i-1])
    
    price_above_kama = close > kama_12h
    price_below_kama = close < kama_12h
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 30  # warmup for KAMA
    
    for i in range(start_idx, n):
        if (np.isnan(kama_trend_aligned[i]) or 
            np.isnan(vol_ma[i]) or
            np.isnan(kama_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA(12h), 1d KAMA trending up, volume spike
            if price_above_kama[i] and kama_trend_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA(12h), 1d KAMA trending down, volume spike
            elif price_below_kama[i] and not kama_trend_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses below KAMA(12h) or volume dries up
            if price_below_kama[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses above KAMA(12h) or volume dries up
            if price_above_kama[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_KAMA_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0
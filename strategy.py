#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Williams %R Reversal with 12h ADX Trend Filter
# Hypothesis: Williams %R identifies overbought/oversold conditions within 12h trends.
# In bull markets, buy pullbacks in uptrends; in bear markets, sell rallies in downtrends.
# ADX filter ensures we only trade when trend is strong enough (ADX > 25).
# Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag.

name = "4h_williamsr_reversal_12h_adx_filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[1:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    dpi = wilders_smoothing(dm_plus, 14)
    dmi = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dpi / atr
    di_minus = 100 * dmi / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilders_smoothing(dx, 14)
    
    # Align 12h ADX to 4h
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Williams %R(14) on 4h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if required data not available
        if np.isnan(adx_12h_aligned[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R reaches overbought or trend weakens
            if williams_r[i] >= -20 or adx_12h_aligned[i] < 25:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: Williams %R reaches oversold or trend weakens
            if williams_r[i] <= -80 or adx_12h_aligned[i] < 25:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Williams %R reversal in direction of 12h trend (ADX > 25)
            if adx_12h_aligned[i] > 25:  # Strong trend
                if williams_r[i] <= -80:  # Oversold - potential long
                    position = 1
                    signals[i] = 0.25
                elif williams_r[i] >= -20:  # Overbought - potential short
                    position = -1
                    signals[i] = -0.25
    
    return signals
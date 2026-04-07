#!/usr/bin/env python3
"""
6h_keltner_meanrev_volume_v1
Hypothesis: On 6h timeframe, enter long when price touches lower Keltner band (mean reversion) with volume spike and 12h EMA trend filter, enter short when price touches upper Keltner band. Exit on middle band reversion. Uses 1d ADX regime filter to avoid ranging markets. Designed for 15-30 trades/year to minimize fee fade while capturing mean reversion in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_keltner_meanrev_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 6h EMA(20) for Keltner middle
    ema_20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Calculate 6h ATR(10) for Keltner width
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10 = pd.Series(tr).ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # Keltner Bands
    mult = 1.5
    upper = ema_20 + mult * atr_10
    lower = ema_20 - mult * atr_10
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Calculate 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 1d ADX(14) for regime filter (avoid ranging markets)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr_1d
    di_minus = 100 * dm_minus_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(ema_20[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: volume > 1.5x average
        vol_spike = volume[i] > 1.5 * vol_ma[i]
        
        # Regime filter: avoid ranging markets (ADX < 20)
        trending = adx_aligned[i] > 20
        
        if position == 1:  # Long position
            # Exit: price crosses above middle band (mean reversion complete)
            if close[i] > ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses below middle band (mean reversion complete)
            if close[i] < ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_spike and trending:
                # Long: price touches lower Keltner band with uptrend
                if low[i] <= lower[i] and close[i] > ema_50_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price touches upper Keltner band with downtrend
                elif high[i] >= upper[i] and close[i] < ema_50_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
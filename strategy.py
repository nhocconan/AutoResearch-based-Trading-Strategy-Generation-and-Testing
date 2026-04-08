#!/usr/bin/env python3
"""
1h Volume Spike with 4h Trend Filter and 1d ADX Confirmation
Hypothesis: In trending markets (4h EMA alignment + 1d ADX > 25), volume spikes on 1h signal
continuation of the trend. Works in bull/bear by requiring trend alignment before entry.
Volume spike = current 1h volume > 2.0 x 20-period average. Targets 15-35 trades/year.
"""

name = "1h_volume_spike_4h_trend_1d_adx_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend (EMA alignment) - call ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Get 1d data for ADX - call ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20 EMA on 4h
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 50 EMA on 4h
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 14-period ADX for 1d
    # True Range
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    
    # Directional Movement
    dm_plus_1d = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                          np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus_1d = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                           np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus_1d = np.concatenate([[0], dm_plus_1d])
    dm_minus_1d = np.concatenate([[0], dm_minus_1d])
    
    # Smoothed values
    tr14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    dm_plus_14_1d = pd.Series(dm_plus_1d).rolling(window=14, min_periods=14).sum().values
    dm_minus_14_1d = pd.Series(dm_minus_1d).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus_1d = 100 * dm_plus_14_1d / tr14_1d
    di_minus_1d = 100 * dm_minus_14_1d / tr14_1d
    
    # DX and ADX
    dx_1d = 100 * np.abs(di_plus_1d - di_minus_1d) / (di_plus_1d + di_minus_1d)
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike detector: current volume > 2.0 x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_4h[i]) or np.isnan(ema50_4h[i]) or 
            np.isnan(adx_1d[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 4h values for current 1h bar
        ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)[i]
        ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)[i]
        
        # Get aligned 1d ADX for current 1h bar
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)[i]
        
        # Trend filter: 4h EMA alignment (bullish: EMA20 > EMA50, bearish: EMA20 < EMA50)
        ema_bullish = ema20_4h_aligned > ema50_4h_aligned
        ema_bearish = ema20_4h_aligned < ema50_4h_aligned
        
        # Regime filter: only trade in strong trending markets on daily
        strong_trend_1d = adx_1d_aligned > 25
        
        if position == 1:  # Long position
            # Exit: trend breaks OR volume spike fades
            if not (ema_bullish and strong_trend_1d):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: trend breaks OR volume spike fades
            if not (ema_bearish and strong_trend_1d):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Only trade with volume spike and both trend filters aligned
            if volume_spike[i] and ema_bullish and strong_trend_1d:
                position = 1
                signals[i] = 0.20
            elif volume_spike[i] and ema_bearish and strong_trend_1d:
                position = -1
                signals[i] = -0.20
    
    return signals
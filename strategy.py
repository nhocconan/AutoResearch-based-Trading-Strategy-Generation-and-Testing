#!/usr/bin/env python3
"""
Hypothesis: 1-day price crosses the 50-day EMA with weekly ADX > 25 and volume above 1.5x 20-day average.
Trades in the direction of the weekly trend (ADX-based) to avoid counter-trend whipsaws.
Designed for low frequency (~10-25 trades/year) to minimize fee drag and work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 50-day EMA on daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate weekly ADX (14-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / (tr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_1w = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-day average volume on daily data
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Get aligned indicators
        ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)[i]
        adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)[i]
        vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)[i]
        
        # Check for NaN values
        if (np.isnan(ema_50_aligned) or np.isnan(adx_1w_aligned) or 
            np.isnan(vol_ma_20_aligned)):
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma_20_aligned
        
        # ADX trend filter (> 25)
        trend_filter = adx_1w_aligned > 25
        
        if position == 0:  # No position - look for entries
            if volume_confirm and trend_filter:
                # Long: price crosses above EMA50
                if close[i] > ema_50_aligned and close[i-1] <= ema_50_aligned:
                    position = 1
                    signals[i] = position_size
                # Short: price crosses below EMA50
                elif close[i] < ema_50_aligned and close[i-1] >= ema_50_aligned:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:  # Long position - exit when price crosses below EMA50
            if close[i] < ema_50_aligned and close[i-1] >= ema_50_aligned:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit when price crosses above EMA50
            if close[i] > ema_50_aligned and close[i-1] <= ema_50_aligned:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_EMA50_1wADX_Volume"
timeframe = "1d"
leverage = 1.0
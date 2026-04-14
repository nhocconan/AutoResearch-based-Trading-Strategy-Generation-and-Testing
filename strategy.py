# 42357
#!/usr/bin/env python3
"""
Hypothesis: 1d price crossing above/below 1-week 50-period EMA with volume above 1.5x 30-period average and 1-week ADX > 25.
Trades in direction of weekly trend to avoid counter-trend whipsaws. Uses EMA for smooth trend and ADX for trend strength.
Targets 15-25 trades/year per symbol (60-100 total over 4 years).
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
    
    # Calculate weekly EMA (50-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate weekly ADX (14-period)
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
    
    # Calculate 30-period average volume
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(60, n):
        # Get aligned indicators
        ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)[i]
        adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)[i]
        vol_ma_30_aligned = vol_ma_30[i]  # already LTF
        
        # Check for NaN values
        if (np.isnan(ema_50_aligned) or np.isnan(adx_1w_aligned) or 
            np.isnan(vol_ma_30_aligned)):
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma_30_aligned
        
        # ADX trend filter (> 25)
        trend_filter = adx_1w_aligned > 25
        
        if position == 0:  # No position - look for entries
            if volume_confirm and trend_filter:
                # Long: price crosses above EMA
                if close[i] > ema_50_aligned and close[i-1] <= ema_50_aligned:
                    position = 1
                    signals[i] = position_size
                # Short: price crosses below EMA
                elif close[i] < ema_50_aligned and close[i-1] >= ema_50_aligned:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:  # Long position - exit when price crosses below EMA
            if close[i] < ema_50_aligned and close[i-1] >= ema_50_aligned:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit when price crosses above EMA
            if close[i] > ema_50_aligned and close[i-1] <= ema_50_aligned:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_1wEMA50_1wADX25_Volume1.5x_v1"
timeframe = "1d"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 1-day price action with 1-week trend filter for mean reversion.
Uses 1-day RSI extremes (overbought/oversold) confirmed by 1-day volume spike
and 1-week ADX < 20 (very weak trend) to enter mean-reversion trades.
Targets 30-80 total trades over 4 years (7-20/year) to minimize fee drag.
Works in both bull and bear markets by fading extremes in ranging conditions.
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
    
    # Get 1d data for RSI and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1-day RSI (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1-day volume spike (volume > 1.8x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 1.8)
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1-week ADX (14-period)
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr0 = np.max([high_1w[0] - low_1w[0], np.abs(high_1w[0] - close_1w[0]), np.abs(low_1w[0] - close_1w[0])])
    tr = np.concatenate([[tr0], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    di_plus = 100 * dm_plus_14 / (tr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr_14 + 1e-10)
    
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # ADX < 20 = very weak trend (ideal for mean reversion)
    low_trend = adx < 20
    
    # Align HTF data to 1d timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    low_trend_aligned = align_htf_to_ltf(prices, df_1w, low_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(low_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: RSI extreme + volume spike + very weak trend
        rsi_oversold = rsi_aligned[i] < 25
        rsi_overbought = rsi_aligned[i] > 75
        vol_confirm = vol_spike_aligned[i] > 0.5
        trend_filter = low_trend_aligned[i] > 0.5
        
        long_entry = rsi_oversold and vol_confirm and trend_filter
        short_entry = rsi_overbought and vol_confirm and trend_filter
        
        # Exit when RSI returns to neutral zone (40-60)
        exit_long = position == 1 and rsi_aligned[i] > 40
        exit_short = position == -1 and rsi_aligned[i] < 60
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_rsi_extreme_mean_reversion"
timeframe = "1d"
leverage = 1.0
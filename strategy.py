#!/usr/bin/env python3
"""
Hypothesis: 12h trading based on 1-week RSI extremes with 1-day volume confirmation and ADX trend filter.
In strong trends (ADX > 25), RSI > 70 indicates overbought (short signal) and RSI < 30 indicates oversold (long signal).
Volume confirmation requires 1-day volume > 1.3x 20-period average to ensure conviction.
Designed to capture mean reversion within trends while avoiding choppy markets.
Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 1-day volume spike (volume > 1.3x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume_1d > (vol_ma_20 * 1.3)
    vol_confirm_aligned = align_htf_to_ltf(prices, df_1d, vol_confirm.astype(float))
    
    # Get 1w data for RSI and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1-week RSI (14-period)
    delta = np.diff(close_1w)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # First average gain/loss
    avg_gain = np.zeros_like(close_1w)
    avg_loss = np.zeros_like(close_1w)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    # Wilder's smoothing
    for i in range(15, len(close_1w)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Calculate 1-week ADX (14-period)
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.max([high_1w[0] - low_1w[0], np.abs(high_1w[0] - close_1w[0]), np.abs(low_1w[0] - close_1w[0])])], 
                        np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    
    # Smoothed values
    tr_14 = np.zeros_like(tr)
    dm_plus_14 = np.zeros_like(dm_plus)
    dm_minus_14 = np.zeros_like(dm_minus)
    
    tr_14[13] = np.sum(tr[0:14])
    dm_plus_14[13] = np.sum(dm_plus[0:14])
    dm_minus_14[13] = np.sum(dm_minus[0:14])
    
    for i in range(14, len(tr)):
        tr_14[i] = tr_14[i-1] - (tr_14[i-1] / 14) + tr[i]
        dm_plus_14[i] = dm_plus_14[i-1] - (dm_plus_14[i-1] / 14) + dm_plus[i]
        dm_minus_14[i] = dm_minus_14[i-1] - (dm_minus_14[i-1] / 14) + dm_minus[i]
    
    # Directional Indicators
    plus_di = 100 * dm_plus_14 / tr_14
    minus_di = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx_1w = np.zeros_like(dx)
    adx_1w[27] = np.mean(dx[14:28])  # First ADX value
    
    for i in range(28, len(dx)):
        adx_1w[i] = (adx_1w[i-1] * 13 + dx[i]) / 14
    
    # Align HTF indicators to LTF
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(adx_1w_aligned[i]) or 
            np.isnan(vol_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: RSI extremes + volume confirmation + trend filter (ADX > 25)
        rsi_overbought = rsi_1w_aligned[i] > 70
        rsi_oversold = rsi_1w_aligned[i] < 30
        strong_trend = adx_1w_aligned[i] > 25
        vol_ok = vol_confirm_aligned[i] > 0.5
        
        long_entry = rsi_oversold and strong_trend and vol_ok
        short_entry = rsi_overbought and strong_trend and vol_ok
        
        # Exit when RSI returns to neutral zone (40-60)
        exit_long = position == 1 and rsi_1w_aligned[i] > 40
        exit_short = position == -1 and rsi_1w_aligned[i] < 60
        
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

name = "12h_1w_rsi_adx_volume"
timeframe = "12h"
leverage = 1.0
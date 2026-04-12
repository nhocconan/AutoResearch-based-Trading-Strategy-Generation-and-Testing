#!/usr/bin/env python3
"""
6h_1w_1d_Volume_Weighted_RSI_Breakout_v1
Hypothesis: On 6h timeframe, buy when price breaks above weekly VWAP with RSI(14)<30 and volume spike,
sell when price breaks below weekly VWAP with RSI(14)>70 and volume spike. Exit when RSI reverts to 50.
Uses daily trend filter (price > EMA50 for longs, < EMA50 for shorts) to avoid counter-trend trades.
Designed for low trade frequency (15-30/year) by requiring multiple confluence factors.
Works in bull/bear via daily trend filter and mean-reversion exit at RSI=50.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_Volume_Weighted_RSI_Breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY VWAP ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Typical price
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    # VWAP calculation
    vwap_num = np.cumsum(typical_price_1w * volume_1w)
    vwap_den = np.cumsum(volume_1w)
    vwap_1w = np.where(vwap_den != 0, vwap_num / vwap_den, np.nan)
    
    # === DAILY TREND FILTER (EMA50) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # EMA50 calculation
    ema50_1d = np.zeros_like(close_1d)
    ema50_1d[:] = np.nan
    multiplier = 2 / (50 + 1)
    for i in range(len(close_1d)):
        if i == 0:
            ema50_1d[i] = close_1d[i]
        elif not np.isnan(close_1d[i]):
            ema50_1d[i] = (close_1d[i] - ema50_1d[i-1]) * multiplier + ema50_1d[i-1]
        else:
            ema50_1d[i] = ema50_1d[i-1]
    
    # === 6h RSI(14) ===
    # RSI calculation with proper handling
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    rsi = np.full(n, np.nan)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    # Initial values
    if n >= 14:
        avg_gain[13] = np.mean(gain[1:14])
        avg_loss[13] = np.mean(loss[1:14])
        for i in range(14, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
            rs = np.where(avg_loss[i] != 0, avg_gain[i] / avg_loss[i], 0)
            rsi[i] = 100 - (100 / (1 + rs))
    
    # Align weekly VWAP to 6h
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # Align daily EMA50 to 6h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period for 6h = ~5 days) for confirmation
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(vwap_1w_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(rsi[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 2x average
        vol_confirm = volume[i] > 2.0 * vol_avg[i]
        
        # Entry conditions
        long_setup = (close[i] > vwap_1w_aligned[i]) and (rsi[i] < 30) and vol_confirm and (close[i] > ema50_1d_aligned[i])
        short_setup = (close[i] < vwap_1w_aligned[i]) and (rsi[i] > 70) and vol_confirm and (close[i] < ema50_1d_aligned[i])
        
        # Exit conditions: RSI mean reversion to 50
        exit_long = rsi[i] > 50
        exit_short = rsi[i] < 50
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
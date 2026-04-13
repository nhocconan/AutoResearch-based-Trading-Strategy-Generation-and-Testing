#!/usr/bin/env python3
"""
4h KAMA Trend with RSI Filter and Volume Confirmation.
Uses Kaufman Adaptive Moving Average to capture trends, filtered by RSI(14) > 50 for longs and < 50 for shorts.
Volume confirmation ensures breakouts have participation. Works in bull markets via trend following
and in bear markets by capturing short-term rebounds and breakdowns with proper filtering.
Designed for ~25-40 trades/year to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for KAMA calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Kaufman Adaptive Moving Average (KAMA)
    kama_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close_4h, kama_period))
    volatility = np.sum(np.abs(np.diff(close_4h)), axis=1)
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close_4h, np.nan)
    kama[kama_period] = close_4h[kama_period]
    for i in range(kama_period+1, len(close_4h)):
        if not np.isnan(sc[i-kama_period]):
            kama[i] = kama[i-1] + sc[i-kama_period] * (close_4h[i] - kama[i-1])
    
    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_4h, kama)
    
    # Get 1h data for RSI and volume
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 30:
        return np.zeros(n)
    
    close_1h = df_1h['close'].values
    volume_1h = df_1h['volume'].values
    
    # RSI(14)
    rsi_period = 14
    delta = np.diff(close_1h)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align RSI to 1h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1h, rsi)
    
    # Volume confirmation: volume > 1.2x 20-period average
    vol_ma = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume_1h > (vol_ma * 1.2)
    vol_conf_aligned = align_htf_to_ltf(prices, df_1h, vol_conf.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_conf_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: price relative to KAMA + RSI filter + volume confirmation
        long_entry = (close[i] > kama_aligned[i] and 
                      rsi_aligned[i] > 50 and 
                      vol_conf_aligned[i] > 0.5)
        short_entry = (close[i] < kama_aligned[i] and 
                       rsi_aligned[i] < 50 and 
                       vol_conf_aligned[i] > 0.5)
        
        # Exit when price crosses back through KAMA
        exit_long = position == 1 and close[i] <= kama_aligned[i]
        exit_short = position == -1 and close[i] >= kama_aligned[i]
        
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

name = "4h_kama_rsi_volume"
timeframe = "4h"
leverage = 1.0
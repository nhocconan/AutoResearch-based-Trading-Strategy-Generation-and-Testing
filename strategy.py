#!/usr/bin/env python3
# 12h_KAMA_Direction_RSI_MeanReversion_1dTrend
# Hypothesis: 12h KAMA direction filter with 1d RSI mean reversion and volume confirmation.
# Uses KAMA to identify trend direction, RSI for mean-reversion entries, and volume to confirm.
# Works in both bull and bear markets by combining trend filter with counter-trend entries.
# Targets 12-37 trades/year to minimize fee drag on 12h timeframe.

name = "12h_KAMA_Direction_RSI_MeanReversion_1dTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for RSI and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align RSI to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Get 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 12h data for KAMA
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA(10, 2, 30) - ER=10, fast=2, slow=30
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10, prepend=close[:10]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
    # Fix: volatility needs to be rolling sum of 10-period changes
    volatility = pd.Series(np.abs(np.diff(close, prepend=close[0]))).rolling(window=10, min_periods=1).sum().values
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align KAMA to 12h (already same timeframe)
    # But we need to ensure it's properly aligned for signal use
    # KAMA is calculated on 12h data, so no alignment needed
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (10), RSI (14), EMA50 (50), volume MA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        # KAMA direction
        kama_up = close[i] > kama[i]
        kama_down = close[i] < kama[i]
        
        # RSI mean reversion levels
        rsi_oversold = rsi_1d_aligned[i] < 30
        rsi_overbought = rsi_1d_aligned[i] > 70
        
        if position == 0:
            # Long: KAMA up + RSI oversold + volume surge (counter-trend in uptrend or with trend)
            if kama_up and rsi_oversold and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down + RSI overbought + volume surge
            elif kama_down and rsi_overbought and volume_surge:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI overbought or KAMA down
            if rsi_1d_aligned[i] > 70 or not kama_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI oversold or KAMA up
            if rsi_1d_aligned[i] < 30 or not kama_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
# 1d_RSI_MeanReversion_1wTrend
# Hypothesis: Use weekly RSI to determine trend direction and daily RSI for mean reversion entries.
# In weekly uptrend (weekly RSI > 50), go long when daily RSI < 30; exit when daily RSI > 70.
# In weekly downtrend (weekly RSI < 50), go short when daily RSI > 70; exit when daily RSI < 30.
# Adds volume confirmation (volume > 20-period average) to avoid false signals.
# Designed for low frequency (10-30 trades/year) to avoid fee drift. Works in both bull and bear markets
# by aligning with the weekly trend while capturing short-term mean reversion.

name = "1d_RSI_MeanReversion_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def rsi(close, period=14):
    """Calculate Relative Strength Index."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    for i in range(len(close)):
        if i < period:
            avg_gain[i] = np.mean(gain[:i+1]) if i > 0 else 0
            avg_loss[i] = np.mean(loss[:i+1]) if i > 0 else 0
        else:
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi_vals = 100 - (100 / (1 + rs))
    return rsi_vals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly RSI for trend
    rsi_1w = rsi(close_1w, 14)
    
    # Calculate daily RSI for entry
    rsi_1d = rsi(close, 14)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly RSI to daily timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(rsi_1d[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Weekly trend filter: RSI > 50 = uptrend, < 50 = downtrend
        weekly_uptrend = rsi_1w_aligned[i] > 50
        weekly_downtrend = rsi_1w_aligned[i] < 50
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        # Daily RSI signals
        rsi_oversold = rsi_1d[i] < 30
        rsi_overbought = rsi_1d[i] > 70
        
        if position == 0:
            # LONG: weekly uptrend, daily RSI oversold, volume confirmation
            if weekly_uptrend and rsi_oversold and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: weekly downtrend, daily RSI overbought, volume confirmation
            elif weekly_downtrend and rsi_overbought and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: daily RSI overbought or weekly trend turns down
            if rsi_overbought or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: daily RSI oversold or weekly trend turns up
            if rsi_oversold or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
"""
Hypothesis: 1h RSI with 4h RSI trend filter and 1d volume confirmation.
In 2025's bear/range market, mean reversion on oversold/overbought RSI works when aligned with higher timeframe momentum.
Uses 4h RSI > 50 for long bias, < 50 for short bias to avoid counter-trend trades.
1d volume filter ensures trades occur during active market participation.
Target: 20-30 trades/year per symbol to minimize fee drag in challenging 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def rsi(close, length):
    """Relative Strength Index"""
    if length <= 0:
        return np.full_like(close, np.nan)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    # Wilder's smoothing
    avg_gain[length-1] = np.mean(gain[:length])
    avg_loss[length-1] = np.mean(loss[:length])
    
    for i in range(length, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
        avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for RSI trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1h RSI(14)
    rsi_1h = rsi(close, 14)
    
    # Calculate 4h RSI(14) for trend filter
    close_4h = df_4h['close'].values
    rsi_4h = rsi(close_4h, 14)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Calculate 1d volume MA(20) for participation filter
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Warmup: need RSI values
    start_idx = 14
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi_1h[i]) or 
            np.isnan(rsi_4h_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        rsi_now = rsi_1h[i]
        rsi_4h_now = rsi_4h_aligned[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        vol_now = volume[i]
        
        # Volume filter: volume > 1.2x daily average
        vol_filter = vol_now > 1.2 * vol_ma
        
        if position == 0:
            # Long: RSI oversold (<30) with 4h bullish bias (>50) and volume
            if rsi_now < 30 and rsi_4h_now > 50 and vol_filter:
                signals[i] = size
                position = 1
            # Short: RSI overbought (>70) with 4h bearish bias (<50) and volume
            elif rsi_now > 70 and rsi_4h_now < 50 and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral (50) or 4h momentum fails
            if rsi_now >= 50 or rsi_4h_now < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI returns to neutral (50) or 4h momentum fails
            if rsi_now <= 50 or rsi_4h_now > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_RSI14_4hRSITrend_1dVolumeFilter"
timeframe = "1h"
leverage = 1.0
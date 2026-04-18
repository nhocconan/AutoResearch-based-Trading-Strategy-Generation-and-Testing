#!/usr/bin/env python3
"""
4h_4H_RSI_Volume_Breakout_v1
Strategy: 4h RSI overbought/oversold with volume spike confirmation and 1D trend filter.
Long: RSI < 30 + volume spike + price > 1D EMA200 (uptrend).
Short: RSI > 70 + volume spike + price < 1D EMA200 (downtrend).
Designed for 4h timeframe: ~20-30 trades/year per symbol (80-120 total over 4 years).
Works in bull/bear via trend filter and mean-reversion RSI logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    
    # Daily EMA200 for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align daily EMA200 to 4h timeframe
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # RSI calculation (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike detection (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma_20  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # need enough for EMA200 and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema_200_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = close[i] > ema_200_aligned[i]
        downtrend = close[i] < ema_200_aligned[i]
        
        if position == 0:
            # Long: oversold RSI + volume spike + uptrend
            if rsi[i] < 30 and vol_spike[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: overbought RSI + volume spike + downtrend
            elif rsi[i] > 70 and vol_spike[i] and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI > 50 or trend change
            if rsi[i] > 50 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI < 50 or trend change
            if rsi[i] < 50 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_4H_RSI_Volume_Breakout_v1"
timeframe = "4h"
leverage = 1.0
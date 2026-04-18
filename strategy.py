#!/usr/bin/env python3
"""
1h_4h1d_Momentum_Confluence
Hypothesis: Combines 4h RSI trend filter with 1d volume surge and 1h momentum breakout.
Uses 4h for trend direction (avoiding counter-trend trades), 1d for institutional volume confirmation,
and 1h for precise entry timing. Targets 15-30 trades/year by requiring confluence of three filters.
Works in bull markets via momentum continuation and in bear markets via mean-reversion bounces
off institutional volume zones.
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
    
    # Get 4h data for RSI trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h RSI(14)
    def rsi(arr, period=14):
        delta = np.diff(arr)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(arr)
        avg_loss = np.zeros_like(arr)
        if len(arr) > period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            for i in range(period+1, len(arr)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        return 100 - (100 / (1 + rs))
    
    rsi_4h = rsi(close_4h, 14)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Get 1d data for volume surge filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume surge (current volume > 2x 20-day average)
    vol_avg_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_avg_1d[i] = np.mean(volume_1d[i-20:i])
    volume_surge_1d = volume_1d > (vol_avg_1d * 2.0)
    volume_surge_aligned = align_htf_to_ltf(prices, df_1d, volume_surge_1d)
    
    # 1h momentum: price > 10-period EMA
    ema10 = np.full(n, np.nan)
    if n >= 10:
        ema10[9] = np.mean(close[0:10])
        alpha = 2 / (10 + 1)
        for i in range(10, n):
            ema10[i] = close[i] * alpha + ema10[i-1] * (1 - alpha)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20, 10)  # RSI needs 14+ for stability, volume needs 20
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(volume_surge_aligned[i]) or 
            np.isnan(ema10[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 4h RSI > 50 (uptrend) + 1d volume surge + 1h price > EMA10
            if (rsi_4h_aligned[i] > 50 and volume_surge_aligned[i] and 
                close[i] > ema10[i]):
                signals[i] = 0.20
                position = 1
            # Short: 4h RSI < 50 (downtrend) + 1d volume surge + 1h price < EMA10
            elif (rsi_4h_aligned[i] < 50 and volume_surge_aligned[i] and 
                  close[i] < ema10[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: 4h RSI < 40 or loss of 1h momentum
            if (rsi_4h_aligned[i] < 40 or close[i] < ema10[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: 4h RSI > 60 or loss of 1h momentum
            if (rsi_4h_aligned[i] > 60 or close[i] > ema10[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h1d_Momentum_Confluence"
timeframe = "1h"
leverage = 1.0
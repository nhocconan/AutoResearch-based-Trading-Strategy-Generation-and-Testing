#!/usr/bin/env python3

"""
Hypothesis: 1h momentum with 4h trend filter and volume confirmation.
Uses RSI(14) momentum on 1h for entry timing, 4h EMA(50) for trend direction,
and volume spike confirmation to avoid false breakouts. Designed for 15-37
trades/year per symbol to minimize fee drag while capturing momentum moves.
Works in both bull and bear markets by following the 4h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI > 60 (momentum), above 4h EMA (trend), volume spike
            if (rsi[i] > 60 and
                close[i] > ema_50_4h_aligned[i] and
                volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI < 40 (momentum), below 4h EMA (trend), volume spike
            elif (rsi[i] < 40 and
                  close[i] < ema_50_4h_aligned[i] and
                  volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: RSI mean reversion or trend change
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI < 50 or below 4h EMA
                if rsi[i] < 50 or close[i] < ema_50_4h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI > 50 or above 4h EMA
                if rsi[i] > 50 or close[i] > ema_50_4h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_RSI_Momentum_4hEMA_Trend_Volume"
timeframe = "1h"
leverage = 1.0
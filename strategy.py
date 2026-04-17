#!/usr/bin/env python3
"""
4h_KAMA_Direction_1dRSI_TrendFilter
Strategy: 4h KAMA direction + 1d RSI > 50 (bullish) or < 50 (bearish) + volume confirmation.
Long: KAMA rising + RSI > 50 + volume > 1.5x 20-period average
Short: KAMA falling + RSI < 50 + volume > 1.5x 20-period average
Exit: Opposite KAMA direction
Position size: 0.25
Uses KAMA for adaptive trend, RSI for momentum filter, volume for confirmation.
Works in both bull (trend follows) and bear (filters counter-trend).
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
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate KAMA (adaptive moving average) on 4h close
    # ER = |close - close[10]| / sum(|close - close[1]| over 10 periods)
    # SSC = [ER * (fastest - slowest) + slowest]^2
    # KAMA[i] = KAMA[i-1] + SSC * (close[i] - KAMA[i-1])
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # |close - close[10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=1)  # sum of 1-period changes
    # Pad arrays to match length
    change_padded = np.concatenate([np.full(9, np.nan), change])
    volatility_padded = np.concatenate([np.full(9, np.nan), volatility[9:]])
    
    er = np.where(volatility_padded != 0, change_padded / volatility_padded, 0)
    # SSC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    ssc = np.square(er * (fast_sc - slow_sc) + slow_sc)
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # start at first valid point
    for i in range(10, n):
        kama[i] = kama[i-1] + ssc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14) on 1d
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    avg_gain[13] = np.mean(gain[1:14]) if len(gain) >= 14 else np.nan
    avg_loss[13] = np.mean(loss[1:14]) if len(loss) >= 14 else np.nan
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad RSI to match 1d length
    rsi_padded = np.concatenate([np.full(14, np.nan), rsi])
    
    # Calculate 20-period volume average on 1d
    volume_ma20 = np.full_like(volume_1d, np.nan)
    for i in range(19, len(volume_1d)):
        volume_ma20[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align indicators to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_padded)
    volume_ma20_aligned = align_htf_to_ltf(prices, df_1d, volume_ma20)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(30, n):  # warmup for indicators
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(volume_ma20_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: rising if current > previous, falling if current < previous
        kama_rising = kama_aligned[i] > kama_aligned[i-1]
        kama_falling = kama_aligned[i] < kama_aligned[i-1]
        
        # RSI filter: > 50 bullish, < 50 bearish
        rsi_bullish = rsi_aligned[i] > 50
        rsi_bearish = rsi_aligned[i] < 50
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_filter = volume_1d_aligned[i] > (1.5 * volume_ma20_aligned[i])
        
        if position == 0:
            # Long: KAMA rising + RSI > 50 + volume
            if kama_rising and rsi_bullish and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling + RSI < 50 + volume
            elif kama_falling and rsi_bearish and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA starts falling
            if kama_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA starts rising
            if kama_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Direction_1dRSI_TrendFilter"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_RSI_Filter
Strategy: 4h KAMA direction with RSI filter and volume confirmation.
Long: KAMA rising + RSI > 50 + volume > 1.5x 20-period average
Short: KAMA falling + RSI < 50 + volume > 1.5x 20-period average
Exit: KAMA direction reversal or RSI crosses 50
Position size: 0.25
Designed to capture trending moves while avoiding chop.
Timeframe: 4h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER = |net change| / sum(|changes|)
    # SC = [ER * (fastest - slowest) + slowest]^2
    # KAMA = prevKAMA + SC * (price - prevKAMA)
    kama_period = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    
    # Net change over kama_period
    net_change = np.abs(np.subtract(close[kama_period:], close[:-kama_period]))
    net_change = np.concatenate([np.full(kama_period, np.nan), net_change])
    
    # Sum of absolute changes over kama_period
    sum_abs_change = np.convolve(abs_change, np.ones(kama_period), mode='full')[:len(close)]
    sum_abs_change = np.concatenate([np.full(kama_period-1, np.nan), sum_abs_change[kama_period-1:]])
    
    # Efficiency Ratio
    er = np.where(sum_abs_change != 0, net_change / sum_abs_change, 0)
    
    # Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI (14-period)
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Initial average gain/loss
    avg_gain = np.convolve(gain, np.ones(rsi_period)/rsi_period, mode='full')[:len(close)]
    avg_loss = np.convolve(loss, np.ones(rsi_period)/rsi_period, mode='full')[:len(close)]
    avg_gain = np.concatenate([np.full(rsi_period-1, np.nan), avg_gain[rsi_period-1:]])
    avg_loss = np.concatenate([np.full(rsi_period-1, np.nan), avg_loss[rsi_period-1:]])
    
    # Smoothed average
    for i in range(rsi_period, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1d trend (close > open = uptrend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    trend_1d = (df_1d['close'] > df_1d['open']).astype(float).values  # 1 for up, 0 for down
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Get 4h volume average (20-period)
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    volume_ma20_4h = np.convolve(volume_4h, np.ones(20)/20, mode='full')[:len(volume_4h)]
    volume_ma20_4h = np.concatenate([np.full(19, np.nan), volume_ma20_4h[19:]])
    volume_ma20_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ma20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Start from sufficient warmup
    start_idx = max(kama_period*2, rsi_period*2, 20)
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(trend_1d_aligned[i]) or 
            np.isnan(volume_ma20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 4h volume
        vol_4h_current = align_htf_to_ltf(prices, df_4h, volume_4h)[i]
        volume_filter = vol_4h_current > (1.5 * volume_ma20_4h_aligned[i])
        
        # KAMA direction
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # RSI conditions
        rsi_above_50 = rsi[i] > 50
        rsi_below_50 = rsi[i] < 50
        
        # Entry signals
        if position == 0:
            # Long: KAMA rising + RSI > 50 + volume filter + 1d uptrend
            if kama_rising and rsi_above_50 and volume_filter and trend_1d_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling + RSI < 50 + volume filter + 1d downtrend
            elif kama_falling and rsi_below_50 and volume_filter and trend_1d_aligned[i] < 0.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA falling or RSI < 50
            if not kama_rising or not rsi_above_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA rising or RSI > 50
            if not kama_falling or not rsi_below_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Trend_With_RSI_Filter"
timeframe = "4h"
leverage = 1.0
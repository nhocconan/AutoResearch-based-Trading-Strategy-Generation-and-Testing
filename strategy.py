#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend with 1d RSI filter and 12h volume confirmation.
# Long when KAMA trend is up (price > KAMA) AND 1d RSI > 50 (bullish momentum) AND 12h volume > 1.3x 20-period average.
# Short when KAMA trend is down (price < KAMA) AND 1d RSI < 50 (bearish momentum) AND 12h volume > 1.3x 20-period average.
# Exit when price crosses KAMA in the opposite direction.
# Uses KAMA for adaptive trend following, RSI for momentum filter, volume for confirmation.
# Target: 60-120 total trades over 4 years (15-30/year) for low fee drift.

name = "12h_KAMA_1dRSI_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h KAMA (adaptive moving average)
    def calculate_kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if len(change.shape) > 1 else np.sum(np.abs(np.diff(close, prepend=close[0])))
        # Vectorized volatility calculation
        volatility = np.convolve(np.abs(np.diff(close, prepend=close[0])), np.ones(length), 'same') / length
        volatility[0] = np.abs(close[0] - close[0])  # Avoid division by zero
        er = np.where(volatility != 0, change / volatility, 0)
        
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # KAMA calculation
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, 10, 2, 30)
    
    # 12h volume filter: current volume > 1.3x 20-period average
    vol_ma20 = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_ma20[:10] = volume[:10].mean() if len(volume) >= 10 else volume[0]
    vol_ma20[-10:] = volume[-10:].mean() if len(volume) >= 10 else volume[-1]
    volume_filter = volume > (1.3 * vol_ma20)
    
    # 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate RSI (14-period) on 1d data
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
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
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = 50  # Neutral before enough data
    
    # Align 1d RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(volume_filter[i]) or np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > KAMA, RSI > 50, volume spike
            long_cond = (close[i] > kama[i]) and (rsi_aligned[i] > 50) and volume_filter[i]
            # Short conditions: price < KAMA, RSI < 50, volume spike
            short_cond = (close[i] < kama[i]) and (rsi_aligned[i] < 50) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
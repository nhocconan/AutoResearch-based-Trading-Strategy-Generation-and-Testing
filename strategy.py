#!/usr/bin/env python3
# Hypothesis: 4h KAMA trend with 1d RSI momentum filter and volume confirmation. 
# Uses KAMA (adaptive moving average) to capture trend direction while reducing whipsaw in choppy markets.
# Enters long when KAMA turns up (bullish) AND 1d RSI > 55 (momentum) AND volume > 1.5x average.
# Enters short when KAMA turns down (bearish) AND 1d RSI < 45 (weak momentum) AND volume > 1.5x average.
# Exits when KAMA direction reverses or volume drops below average.
# Target: 80-150 total trades over 4 years (20-38/year) with size 0.25.

name = "4h_KAMA_RSI_Volume_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (adaptive moving average) - trend indicator
    def calculate_kama(close_array, er_length=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close_array, prepend=close_array[0]))
        volatility = np.sum(np.abs(np.diff(close_array)), axis=0) if len(close_array.shape) > 0 else np.abs(np.diff(close_array))
        # For 1D array: volatility over er_length period
        volatility_rolling = np.zeros_like(close_array)
        for i in range(len(close_array)):
            if i < er_length:
                volatility_rolling[i] = np.sum(np.abs(np.diff(close_array[max(0, i-er_length+1):i+1]))) if i > 0 else 0
            else:
                volatility_rolling[i] = np.sum(np.abs(np.diff(close_array[i-er_length+1:i+1])))
        
        er = np.where(volatility_rolling != 0, change / volatility_rolling, 0)
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama = np.zeros_like(close_array)
        kama[0] = close_array[0]
        for i in range(1, len(close_array)):
            kama[i] = kama[i-1] + sc[i] * (close_array[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30)
    kama_up = kama > np.roll(kama, 1)  # KAMA rising
    kama_down = kama < np.roll(kama, 1)  # KAMA falling
    
    # 1-day RSI for momentum filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = np.where(avg_loss == 0, 100, rsi_1d)
    rsi_1d = np.where(avg_gain == 0, 0, rsi_1d)
    
    rsi_high = rsi_1d > 55  # bullish momentum
    rsi_low = rsi_1d < 45   # bearish momentum
    
    rsi_high_aligned = align_htf_to_ltf(prices, df_1d, rsi_high)
    rsi_low_aligned = align_htf_to_ltf(prices, df_1d, rsi_low)
    
    # Volume confirmation
    volume_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            volume_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i > 0 else volume[0]
        else:
            volume_ma[i] = np.mean(volume[i-19:i+1])
    volume_high = volume > 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_up[i]) or np.isnan(kama_down[i]) or
            np.isnan(rsi_high_aligned[i]) or np.isnan(rsi_low_aligned[i]) or
            np.isnan(volume_high[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA up + RSI > 55 + high volume
            if kama_up[i] and rsi_high_aligned[i] and volume_high[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA down + RSI < 45 + high volume
            elif kama_down[i] and rsi_low_aligned[i] and volume_high[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA turns down OR volume drops
            if kama_down[i] or not volume_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA turns up OR volume drops
            if kama_up[i] or not volume_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
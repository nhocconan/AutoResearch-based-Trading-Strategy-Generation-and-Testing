#!/usr/bin/env python3
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
    
    # === 1d KAMA (2-period ER, 30-period SMA) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close_1d, k=10, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d, k=1, prepend=close_1d[0])), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama = np.copy(close_1d)
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # === 1d RSI (14-period) ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    period = 14
    for i in range(len(gain)):
        if i < period:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * (i-1) + gain[i]) / i
                avg_loss[i] = (avg_loss[i-1] * (i-1) + loss[i]) / i
        else:
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[avg_loss == 0] = 100
    
    # === Align indicators to 6h timeframe ===
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 6h Donchian Channel (20-period) ===
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    for i in range(len(high)):
        if i >= 19:
            highest_high[i] = np.max(high[i-19:i+1])
            lowest_low[i] = np.min(low[i-19:i+1])
        elif i > 0:
            highest_high[i] = np.max(high[max(0, i-9):i+1])
            lowest_low[i] = np.min(low[max(0, i-9):i+1])
        else:
            highest_high[i] = high[i]
            lowest_low[i] = low[i]
    
    # === 6h Volume confirmation ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[i]
    
    vol_confirm = volume > vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat AND volume confirmation
        if position == 0:
            # Long: Price breaks above Donchian high + price > KAMA + RSI < 50 (pullback in uptrend) + volume confirmation
            if (close[i] > highest_high[i] and 
                close[i] > kama_aligned[i] and 
                rsi_1d_aligned[i] < 50 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Price breaks below Donchian low + price < KAMA + RSI > 50 (pullback in downtrend) + volume confirmation
            elif (close[i] < lowest_low[i] and 
                  close[i] < kama_aligned[i] and 
                  rsi_1d_aligned[i] > 50 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Price crosses below KAMA OR RSI > 70 (overbought)
            if (close[i] < kama_aligned[i] or 
                rsi_1d_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above KAMA OR RSI < 30 (oversold)
            if (close[i] > kama_aligned[i] or 
                rsi_1d_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_KAMA_Donchian_RSI_Breakout_Pullback_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0
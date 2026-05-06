#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day KAMA trend direction with RSI(14) mean reversion and volume confirmation
# Long when 1-day KAMA is rising (bullish trend) and RSI(14) < 30 (oversold) with volume > 1.3x average
# Short when 1-day KAMA is falling (bearish trend) and RSI(14) > 70 (overbought) with volume > 1.3x average
# Uses KAMA for adaptive trend following, RSI for mean reversion entries, volume for confirmation
# Designed to work in bull markets via buying dips in uptrends and in bear markets via selling rallies in downtrends
# Target: 20-30 trades per year (80-120 over 4 years) with 0.25 position sizing

name = "4h_1dKAMA_RSI_MeanReversion_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day KAMA (adaptive moving average)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Efficiency Ratio (ER) for KAMA
    price_change = np.abs(np.diff(df_1d['close'].values))
    volatility = np.sum(np.abs(np.diff(df_1d['close'].values)), axis=0) if len(df_1d) > 1 else np.array([0])
    # Corrected ER calculation
    change = np.abs(df_1d['close'].values - np.roll(df_1d['close'].values, 1))
    change[0] = 0
    volatility_sum = np.abs(np.diff(df_1d['close'].values))
    volatility_sum = np.concatenate([[0], volatility_sum])
    
    # Rolling window for ER calculation
    er = np.zeros_like(df_1d['close'].values)
    for i in range(len(df_1d)):
        if i >= 10:
            direction = np.abs(df_1d['close'].values[i] - df_1d['close'].values[i-10])
            volatility = np.sum(np.abs(np.diff(df_1d['close'].values[i-9:i+1])))
            if volatility > 0:
                er[i] = direction / volatility
            else:
                er[i] = 0
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(df_1d['close'].values)
    kama[0] = df_1d['close'].values[0]
    for i in range(1, len(df_1d)):
        kama[i] = kama[i-1] + sc[i] * (df_1d['close'].values[i] - kama[i-1])
    
    # KAMA direction: rising if current > previous
    kama_rising = kama > np.roll(kama, 1)
    kama_rising[0] = False
    kama_falling = kama < np.roll(kama, 1)
    kama_falling[0] = False
    
    # Align KAMA direction to 4h timeframe
    kama_rising_aligned = align_htf_to_ltf(prices, df_1d, kama_rising.astype(float))
    kama_falling_aligned = align_htf_to_ltf(prices, df_1d, kama_falling.astype(float))
    
    # Calculate RSI(14) on 4h close
    delta = np.diff(close)
    delta = np.concatenate([[0], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    for i in range(len(close)):
        if i < 14:
            if i > 0:
                avg_gain[i] = np.mean(gain[1:i+1]) if i >= 1 else 0
                avg_loss[i] = np.mean(loss[1:i+1]) if i >= 1 else 0
            else:
                avg_gain[i] = 0
                avg_loss[i] = 0
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: >1.3x 20-period average
    vol_ma_20 = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 20:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else 0
    volume_filter = volume > (1.3 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after RSI warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(kama_rising_aligned[i]) or np.isnan(kama_falling_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA rising (uptrend) + RSI oversold + volume confirmation
            if kama_rising_aligned[i] > 0.5 and rsi[i] < 30 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling (downtrend) + RSI overbought + volume confirmation
            elif kama_falling_aligned[i] > 0.5 and rsi[i] > 70 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI overbought or KAMA turns down
            if rsi[i] > 70 or kama_falling_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI oversold or KAMA turns up
            if rsi[i] < 30 or kama_rising_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
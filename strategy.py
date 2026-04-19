#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend filter with RSI mean reversion and volume spike confirmation.
# Long when KAMA turns bullish (price > KAMA), RSI < 40, and volume > 2x average.
# Short when KAMA turns bearish (price < KAMA), RSI > 60, and volume > 2x average.
# Exit when price crosses back over KAMA.
# Uses 12h timeframe with daily KAMA and RSI for trend/momentum alignment.
# Target: 20-40 trades/year per symbol to stay within frequency limits.
name = "12h_KAMA_RSI_VolumeSpike"
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
    
    # Get daily data for KAMA and RSI calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will compute properly below
    # Correct volatility calculation: sum of absolute changes over window
    volatility = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i == 0:
            volatility[i] = 0
        else:
            volatility[i] = np.sum(np.abs(np.diff(close_1d[max(0, i-9):i+1])))  # 10-period
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # first 14 periods
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align KAMA and RSI to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Get 12h average volume for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure KAMA, RSI, and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long entry: price > KAMA (bullish trend), RSI < 40 (oversold), volume spike
            if price > kama_val and rsi_val < 40 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price < KAMA (bearish trend), RSI > 60 (overbought), volume spike
            elif price < kama_val and rsi_val > 60 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below KAMA (trend change)
            if price < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above KAMA (trend change)
            if price > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
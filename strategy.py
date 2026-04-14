#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA with 1d RSI and Volume Filter
# Long when 4h price > KAMA (trend up) AND 1d RSI < 70 (not overbought) AND 1d volume > 1.5x average
# Short when 4h price < KAMA (trend down) AND 1d RSI > 30 (not oversold) AND 1d volume > 1.5x average
# Exit when price crosses back below/above KAMA
# KAMA adapts to market noise, reducing whipsaws in sideways markets
# Target: 25-50 trades per symbol over 4 years (6-12.5/year)

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h KAMA (adaptive moving average)
    # Efficiency Ratio (ER) over 10 periods
    change_10 = np.abs(np.diff(close, n=10, prepend=close[0]))
    volatility_10 = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if len(close) >= 10 else np.zeros_like(close)
    # Fix volatility calculation for 10-period sum
    volatility_10 = np.array([np.sum(np.abs(np.diff(close[max(0,i-9):i+1]))) if i >= 9 else 0.0 for i in range(len(close))])
    er = np.where(volatility_10 != 0, change_10 / volatility_10, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 1d RSI (14-period)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate 1d average volume (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_4h, kama)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30  # for KAMA and RSI calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1d_current = vol_1d[i] if i < len(vol_1d) else vol_1d[-1]
        
        if position == 0:
            # Long setup: price above KAMA, RSI not overbought, volume spike
            if (price > kama_aligned[i] and 
                rsi_1d_aligned[i] < 70 and 
                vol_1d_current > 1.5 * vol_ma_1d_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: price below KAMA, RSI not oversold, volume spike
            elif (price < kama_aligned[i] and 
                  rsi_1d_aligned[i] > 30 and 
                  vol_1d_current > 1.5 * vol_ma_1d_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below KAMA
            if price < kama_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above KAMA
            if price > kama_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_KAMA_1dRSI_VolumeFilter"
timeframe = "4h"
leverage = 1.0
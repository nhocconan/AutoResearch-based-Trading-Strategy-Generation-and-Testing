#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_KAMA_Direction_RSI_Filter_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for KAMA trend filter and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily close
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)
    # Handle first 9 values where we don't have 10-period difference
    change = np.concatenate([np.full(9, np.nan), change])
    volatility = np.concatenate([np.full(9, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # Start after first 9 periods
    for i in range(10, len(close_1d)):
        if np.isnan(kama[i-1]) or np.isnan(sc[i]):
            kama[i] = close_1d[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_1d = kama
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate RSI(14) on daily close
    delta = np.diff(close_1d, n=1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # First average gain/loss
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    avg_gain[14] = np.mean(gain[1:15])  # indices 1 to 14
    avg_loss[14] = np.mean(loss[1:15])
    for i in range(15, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_1d = rsi
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 20-period volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 for volume average
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_val = kama_1d_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Enter long: Price above KAMA AND RSI > 50 AND volume spike
            if close[i] > kama_val and rsi_val > 50 and vol > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Price below KAMA AND RSI < 50 AND volume spike
            elif close[i] < kama_val and rsi_val < 50 and vol > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price below KAMA OR RSI < 40
            if close[i] < kama_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price above KAMA OR RSI > 60
            if close[i] > kama_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend with RSI momentum filter and volume spike
# Uses Kaufman's Adaptive Moving Average (KAMA) to capture trend direction,
# confirmed by RSI momentum and volume spikes. Designed for low-frequency
# trading (12-37 trades/year) to minimize fee drag and improve robustness
# across bull and bear markets.

name = "12h_KAMA_RSI_Volume"
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
    
    # Get 1d data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA ( Kaufman's Adaptive Moving Average )
    # ER (Efficiency Ratio) = |close - close[10]| / sum(|close - close[-1]|) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=1)  # sum of 1-period changes
    # Pad volatility to match length
    volatility_padded = np.concatenate([np.full(9, np.nan), volatility])
    er = np.full_like(close_1d, np.nan, dtype=float)
    er[10:] = change[9:] / volatility_padded[10:]
    er[np.isnan(er)] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if np.isnan(kama[i-1]):
            kama[i] = close_1d[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI(14) on 1d
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    avg_gain[14] = np.nanmean(gain[1:15])
    avg_loss[14] = np.nanmean(loss[1:15])
    
    for i in range(15, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume spike (2.0x 20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above KAMA, RSI > 50, and volume spike
            if (close[i] > kama_aligned[i] and 
                rsi_aligned[i] > 50 and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA, RSI < 50, and volume spike
            elif (close[i] < kama_aligned[i] and 
                  rsi_aligned[i] < 50 and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below KAMA or RSI < 40
            if (close[i] < kama_aligned[i] or 
                rsi_aligned[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above KAMA or RSI > 60
            if (close[i] > kama_aligned[i] or 
                rsi_aligned[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
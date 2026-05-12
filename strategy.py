#!/usr/bin/env python3
name = "12h_KAMA_Direction_RSI_Pullback"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data once for trend filter and RSI
    df_1d = get_htf_data(prices, '1d')
    
    # 1d KAMA: Efficiency Ratio and smoothing constant
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder for efficiency ratio calc
    # Correct ER calculation: change / volatility over lookback
    er = np.zeros_like(close_1d)
    lookback = 10
    for i in range(lookback, len(close_1d)):
        if i == lookback:
            change_sum = np.sum(np.abs(np.diff(close_1d[i-lookback:i+1])))
            vol_sum = np.sum(np.abs(np.diff(close_1d[i-lookback:i+1])))
        else:
            change_sum = change_sum - np.abs(close_1d[i-lookback] - close_1d[i-lookback-1]) + np.abs(close_1d[i] - close_1d[i-1])
            vol_sum = vol_sum - np.abs(close_1d[i-lookback] - close_1d[i-lookback-1]) + np.abs(close_1d[i] - close_1d[i-1])
        if vol_sum > 0:
            er[i] = change_sum / vol_sum
        else:
            er[i] = 0
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # fast=2, slow=30
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama = np.where(np.arange(len(close_1d)) < 30, np.nan, kama)  # warmup
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # 1d RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > KAMA (uptrend) + RSI pulls back from >70 to <70 + volume spike
            if (close[i] > kama_aligned[i] and 
                rsi[i] < 70 and 
                rsi[i-1] >= 70 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA (downtrend) + RSI bounces from <30 to >30 + volume spike
            elif (close[i] < kama_aligned[i] and 
                  rsi[i] > 30 and 
                  rsi[i-1] <= 30 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI > 70 (overbought)
            if rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI < 30 (oversold)
            if rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
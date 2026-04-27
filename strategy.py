#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA trend and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d KAMA for trend filter
    # KAMA requires efficiency ratio and smoothing constants
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        direction = np.abs(close_1d[i] - close_1d[i-10])
        volatility_sum = np.sum(volatility[i-9:i+1])
        if volatility_sum > 0:
            er[i] = direction / volatility_sum
        else:
            er[i] = 0
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate 1d RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align to 4h timeframe
    kama_4h = align_htf_to_ltf(prices, df_1d, kama)
    rsi_4h = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_4h[i]) or np.isnan(rsi_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Price above KAMA (uptrend) and RSI < 30 (oversold) and volume spike
        # 2. Price crosses above KAMA with RSI < 50 and volume spike
        long_condition = ((close[i] > kama_4h[i] and rsi_4h[i] < 30 and volume_spike[i]) or
                         (close[i] > kama_4h[i] and close[i-1] <= kama_4h[i-1] and 
                          rsi_4h[i] < 50 and volume_spike[i]))
        
        # Short conditions:
        # 1. Price below KAMA (downtrend) and RSI > 70 (overbought) and volume spike
        # 2. Price crosses below KAMA with RSI > 50 and volume spike
        short_condition = ((close[i] < kama_4h[i] and rsi_4h[i] > 70 and volume_spike[i]) or
                          (close[i] < kama_4h[i] and close[i-1] >= kama_4h[i-1] and 
                           rsi_4h[i] > 50 and volume_spike[i]))
        
        if long_condition:
            signals[i] = 0.25
            position = 1
        elif short_condition:
            signals[i] = -0.25
            position = -1
        # Exit conditions: reverse signal or RSI crosses 50
        elif position == 1 and (rsi_4h[i] > 50 or close[i] < kama_4h[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (rsi_4h[i] < 50 or close[i] > kama_4h[i]):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_KAMA_RSI_VolumeSpike_Trend"
timeframe = "4h"
leverage = 1.0
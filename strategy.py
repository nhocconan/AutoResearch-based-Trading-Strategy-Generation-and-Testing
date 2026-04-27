#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for KAMA trend and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER = |Close - Close[10]| / Sum(|Close - Close[1]|, 10)
    # SSC = [ER * (Fastest SC - Slowest SC) + Slowest SC]^2
    # KAMA = KAMA[1] + SSC * (Close - KAMA[1])
    change = np.abs(np.diff(close_1d, n=10))
    volatility = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        volatility[i] = np.sum(np.abs(np.diff(close_1d[i-9:i+1, None], axis=1)))
    ER = np.zeros_like(close_1d)
    ER[10:] = change[10:] / volatility[10:]
    ER[volatility == 0] = 0
    FastSC = 2 / (2 + 1)
    SlowSC = 2 / (30 + 1)
    SSC = (ER * (FastSC - SlowSC) + SlowSC) ** 2
    KAMA = np.zeros_like(close_1d)
    KAMA[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        KAMA[i] = KAMA[i-1] + SSC[i] * (close_1d[i] - KAMA[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    RSI = 100 - (100 / (1 + rs))
    
    # Align KAMA and RSI to 12h timeframe
    KAMA_12h = align_htf_to_ltf(prices, df_1d, KAMA)
    RSI_12h = align_htf_to_ltf(prices, df_1d, RSI)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(KAMA_12h[i]) or np.isnan(RSI_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Price above KAMA with RSI > 50 and volume spike
        long_condition = (close[i] > KAMA_12h[i] and RSI_12h[i] > 50 and volume_spike[i])
        
        # Short conditions:
        # 1. Price below KAMA with RSI < 50 and volume spike
        short_condition = (close[i] < KAMA_12h[i] and RSI_12h[i] < 50 and volume_spike[i])
        
        if long_condition and position != 1:
            signals[i] = 0.25
            position = 1
        elif short_condition and position != -1:
            signals[i] = -0.25
            position = -1
        # Exit conditions: reverse signal
        elif (long_condition and position == -1) or (short_condition and position == 1):
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

name = "12h_KAMA_RSI_VolumeSpike_Trend"
timeframe = "12h"
leverage = 1.0
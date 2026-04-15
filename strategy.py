#!/usr/bin/env python3
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
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Supertrend(10,3) for trend direction
    hl2 = (df_4h['high'] + df_4h['low']) / 2
    tr1 = df_4h['high'] - df_4h['low']
    tr2 = np.abs(df_4h['high'] - np.concatenate([[hl2[0]], hl2[:-1]]))
    tr3 = np.abs(df_4h['low'] - np.concatenate([[hl2[0]], hl2[:-1]]))
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10_4h = pd.Series(tr_4h).ewm(span=10, adjust=False, min_periods=10).mean().values
    upper_band = hl2 + 3 * atr_10_4h
    lower_band = hl2 - 3 * atr_10_4h
    
    supertrend = np.zeros_like(hl2)
    direction = np.ones_like(hl2)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(hl2)):
        if hl2[i] > upper_band[i-1]:
            direction[i] = 1
        elif hl2[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    supertrend_aligned = align_htf_to_ltf(prices, df_4h, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_4h, direction)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily RSI(14) for momentum filter
    delta = pd.Series(df_1d['close'].values).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Calculate 1h Bollinger Bands for entry timing
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or 
            np.isnan(rsi_14_1d_aligned[i]) or np.isnan(sma_20[i]) or np.isnan(std_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = prices.index[i].hour
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 4h Supertrend uptrend (direction = 1)
        # 2. Daily RSI between 30 and 70 (not extreme)
        # 3. Price touches or crosses below lower Bollinger Band (mean reversion entry)
        if (direction_aligned[i] == 1 and
            30 <= rsi_14_1d_aligned[i] <= 70 and
            close[i] <= lower_bb[i]):
            signals[i] = 0.20
            
        # Short conditions:
        # 1. 4h Supertrend downtrend (direction = -1)
        # 2. Daily RSI between 30 and 70 (not extreme)
        # 3. Price touches or crosses above upper Bollinger Band (mean reversion entry)
        elif (direction_aligned[i] == -1 and
              30 <= rsi_14_1d_aligned[i] <= 70 and
              close[i] >= upper_bb[i]):
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_Supertrend4h_RSI1d_BB_v1"
timeframe = "1h"
leverage = 1.0
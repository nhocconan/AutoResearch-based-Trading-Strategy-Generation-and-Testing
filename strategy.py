#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_Trend_Follow
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) for trend direction and RSI for momentum confirmation in the same direction. Enter long when KAMA turns up and RSI > 50; short when KAMA turns down and RSI < 50. This captures momentum in trending markets while avoiding whipsaws in sideways markets. Works in bull (buy dips) and bear (sell rallies) by following the adaptive trend. Target: 20-40 trades/year per symbol.
"""

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d KAMA for trend direction
    close_1d = pd.Series(df_1d['close'].values)
    # Calculate Efficiency Ratio (ER)
    change = abs(close_1d - close_1d.shift(10))
    volatility = abs(close_1d - close_1d.shift(1)).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Calculate KAMA
    kama = np.zeros(len(close_1d))
    kama[0] = close_1d.iloc[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close_1d.iloc[i] - kama[i-1])
    kama = kama
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI(14) for momentum confirmation
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Volume filter: require volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 30  # need 30 for KAMA and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA turning up and RSI > 50 with volume confirmation
            if (i > 0 and kama_1d_aligned[i] > kama_1d_aligned[i-1] and 
                rsi[i] > 50 and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA turning down and RSI < 50 with volume confirmation
            elif (i > 0 and kama_1d_aligned[i] < kama_1d_aligned[i-1] and 
                  rsi[i] < 50 and volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: KAMA turns down or RSI < 40
            if (kama_1d_aligned[i] < kama_1d_aligned[i-1] or rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA turns up or RSI > 60
            if (kama_1d_aligned[i] > kama_1d_aligned[i-1] or rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Direction_RSI_Trend_Follow"
timeframe = "4h"
leverage = 1.0
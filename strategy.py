#!/usr/bin/env python3
"""
12h_1d_KAMA_With_Trend_Filter
Hypothesis: Kaufman Adaptive Moving Average (KAMA) adapts to market noise, providing a dynamic trend filter.
On 12h timeframe, price above/below KAMA with volume confirmation and daily trend filter (200 EMA) captures
trend moves while avoiding whipsaws in chop. Works in both bull and bear markets by following the trend.
Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (10-period ER, 2/30 fast/slow SC)
    close_series = pd.Series(close)
    change = abs(close_series - close_series.shift(10))
    volatility = abs(close_series - close_series.shift(1)).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    sc = sc.fillna(0)
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Daily 200 EMA for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(200, n):
        # Skip if any required data is not ready
        if (np.isnan(kama[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long: price above KAMA, above daily EMA200, with volume expansion
        long_condition = (close[i] > kama[i]) and (close[i] > ema_200_1d_aligned[i]) and volume_expansion[i]
        
        # Short: price below KAMA, below daily EMA200, with volume expansion
        short_condition = (close[i] < kama[i]) and (close[i] < ema_200_1d_aligned[i]) and volume_expansion[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "12h_1d_KAMA_With_Trend_Filter"
timeframe = "12h"
leverage = 1.0
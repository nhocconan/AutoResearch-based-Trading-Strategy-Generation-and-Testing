#!/usr/bin/env python3
"""
1d_1w_KAMA_Trend_With_Volume_Filter
Hypothesis: Kaufman Adaptive Moving Average (KAMA) on daily timeframe adapts to market noise,
reducing whipsaws in ranging markets while capturing trends. Weekly trend filter ensures
alignment with higher-timeframe momentum. Volume expansion confirms institutional interest.
Designed for 1-3 trades per month (12-36/year) to minimize fee drag. Works in bull markets
by riding trends and in bear markets by avoiding false signals during low volatility.
"""

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
    
    # Get daily data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    # Parameters: ER period = 10, Fast SC = 2/(2+1), Slow SC = 2/(30+1)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0)
    # Fix: volatility should be rolling sum
    volatility = pd.Series(close_1d).rolling(window=10, min_periods=1).apply(
        lambda x: np.sum(np.abs(np.diff(x, prepend=x[0]))), raw=True
    ).values
    
    # Avoid division by zero
    er = np.where(volatility > 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if np.isnan(kama[i-1]):
            kama[i] = close_1d[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_20_1w)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.8)
    
    # Align daily KAMA to 1d timeframe (no additional delay needed for KAMA itself)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(sma_20_1w_aligned[i]) or
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Price above KAMA (bullish trend)
        # 2. Weekly close above 20-week SMA (higher-timeframe uptrend)
        # 3. Volume expansion
        kama_bullish = close[i] > kama_aligned[i]
        weekly_uptrend = close[i] > sma_20_1w_aligned[i]
        long_condition = kama_bullish and weekly_uptrend and volume_expansion[i]
        
        # Short conditions:
        # 1. Price below KAMA (bearish trend)
        # 2. Weekly close below 20-week SMA (higher-timeframe downtrend)
        # 3. Volume expansion
        kama_bearish = close[i] < kama_aligned[i]
        weekly_downtrend = close[i] < sma_20_1w_aligned[i]
        short_condition = kama_bearish and weekly_downtrend and volume_expansion[i]
        
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

name = "1d_1w_KAMA_Trend_With_Volume_Filter"
timeframe = "1d"
leverage = 1.0
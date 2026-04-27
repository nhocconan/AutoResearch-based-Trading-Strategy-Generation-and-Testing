#!/usr/bin/env python3
"""
1h_KAMA_Trend_With_1d_Range_Filter
Hypothesis: Use 1h Kaufman Adaptive Moving Average (KAMA) with 1d high/low range filter to enter trend-following positions. 
KAMA adapts to market noise, reducing whipsaws in ranging markets. 
Enter long when price > KAMA and above 1d low + 0.382*range (pullback support in uptrend).
Enter short when price < KAMA and below 1d high - 0.382*range (pullback resistance in downtrend).
Use 1d trend (EMA34) to filter direction. Only trade during 08-20 UTC to avoid low-liquidity hours.
Target: 15-25 trades/year per symbol with tight stops via signal=0 on trend failure.
Works in bull (buy dips) and bear (sell rallies) by trading with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and range
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d range for support/resistance levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    range_1d = high_1d - low_1d
    # Pullback levels: 38.2% retracement of 1d range
    support_1d = low_1d + range_1d * 0.382   # buy zone in uptrend
    resistance_1d = high_1d - range_1d * 0.382  # sell zone in downtrend
    
    support_aligned = align_htf_to_ltf(prices, df_1d, support_1d)
    resistance_aligned = align_htf_to_ltf(prices, df_1d, resistance_1d)
    
    # 1h KAMA (adaptive moving average)
    # Efficiency Ratio: ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(close - np.roll(close, 10))
    change[0:10] = np.nan  # first 10 values invalid
    
    # Volatility: sum of absolute changes over 10 periods
    volatility = np.zeros(n)
    for i in range(n):
        if i < 10:
            volatility[i] = np.nan
        else:
            volatility[i] = np.nansum(np.abs(close[i-9:i+1] - np.roll(close[i-9:i+1], 1)))
    
    # Avoid division by zero
    er = np.zeros(n)
    er[:] = np.nan
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[:] = np.nan
    kama[0] = close[0]  # seed
    
    for i in range(1, n):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume filter: avoid low-volume false signals
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 10 for KAMA, 20 for vol MA
    start_idx = max(20, 10)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(support_aligned[i]) or 
            np.isnan(resistance_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check session
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > KAMA (uptrend) and above 1d support level with volume
            if (close[i] > kama[i] and 
                close[i] > support_aligned[i] and 
                ema34_1d_aligned[i] > kama[i] and  # 1d trend confirms uptrend
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: price < KAMA (downtrend) and below 1d resistance level with volume
            elif (close[i] < kama[i] and 
                  close[i] < resistance_aligned[i] and 
                  ema34_1d_aligned[i] < kama[i] and  # 1d trend confirms downtrend
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below KAMA or 1d trend fails
            if (close[i] <= kama[i] or 
                ema34_1d_aligned[i] <= kama[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above KAMA or 1d trend fails
            if (close[i] >= kama[i] or 
                ema34_1d_aligned[i] >= kama[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_KAMA_Trend_With_1d_Range_Filter"
timeframe = "1h"
leverage = 1.0
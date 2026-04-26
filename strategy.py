#!/usr/bin/env python3
"""
1h_HTF_Trend_With_Volume_Regime_Filter
Hypothesis: On 1h timeframe, capture medium-term trends by requiring alignment between 1h price action, 4h EMA50 trend, and 1d EMA200 trend, with volume regime filter (low volume = choppy, high volume = trending). Enter long when 1h close > 4h EMA50 > 1d EMA200 AND volume > 1.5x 20-period average. Enter short when 1h close < 4h EMA50 < 1d EMA200 AND volume > 1.5x 20-period average. Uses discrete position size 0.20 to limit fee churn. Designed for 15-30 trades/year on 1h by requiring strong HTF alignment and volume confirmation, reducing overtrading while capturing sustained moves in both bull and bear markets.
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
    
    # Get HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for intermediate trend
    close_4h_series = pd.Series(df_4h['close'].values)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d EMA200 for long-term trend
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume regime: >1.5x 20-period average indicates trending conditions
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_trending = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup and volume MA warmup
    start_idx = max(50, 200, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Determine HTF trend alignment
        # Long regime: price above both EMAs AND EMAs in correct order (4h > 1d)
        long_regime = (close[i] > ema_50_4h_aligned[i]) and \
                      (ema_50_4h_aligned[i] > ema_200_1d_aligned[i])
        
        # Short regime: price below both EMAs AND EMAs in correct order (4h < 1d)
        short_regime = (close[i] < ema_50_4h_aligned[i]) and \
                       (ema_50_4h_aligned[i] < ema_200_1d_aligned[i])
        
        if position == 0:
            # Enter long: long regime + volume trending
            if long_regime and volume_trending[i]:
                signals[i] = 0.20
                position = 1
            # Enter short: short regime + volume trending
            elif short_regime and volume_trending[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: regime breaks OR volume drops (choppy conditions)
            if not (long_regime and volume_trending[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: regime breaks OR volume drops (choppy conditions)
            if not (short_regime and volume_trending[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_HTF_Trend_With_Volume_Regime_Filter"
timeframe = "1h"
leverage = 1.0
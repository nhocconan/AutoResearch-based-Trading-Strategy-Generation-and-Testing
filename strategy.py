#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_Volume_Filter
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) as primary trend filter on 1d timeframe.
Enter long when price > KAMA and volume > 1.5x 20-period average; short when price < KAMA and volume > 1.5x average.
Exit when price crosses back below/above KAMA.
KAMA adapts to market noise, reducing whipsaws in choppy markets while capturing trends.
Volume filter ensures participation only during active market conditions.
Target: 15-25 trades/year to minimize fee drag while maintaining edge in both bull and bear markets.
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
    
    # Get weekly data for higher timeframe context (trend confirmation)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate KAMA on daily close prices
    # KAMA parameters: ER period=10, Fast=2, Slow=30
    close_series = pd.Series(close)
    change = abs(close_series.diff(10))
    volatility = close_series.diff().abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    sc = sc.fillna(0)
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for KAMA calculation and volatility
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > KAMA AND weekly trend bullish (close > EMA50) AND volume filter
            long_setup = (close[i] > kama[i]) and \
                         (close[i] > ema_50_1w_aligned[i]) and \
                         volume_filter[i]
            # Short: price < KAMA AND weekly trend bearish (close < EMA50) AND volume filter
            short_setup = (close[i] < kama[i]) and \
                          (close[i] < ema_50_1w_aligned[i]) and \
                          volume_filter[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price crosses back below KAMA OR weekly trend turns bearish
            if (close[i] < kama[i]) or \
               (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price crosses back above KAMA OR weekly trend turns bullish
            if (close[i] > kama[i]) or \
               (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_With_Volume_Filter"
timeframe = "1d"
leverage = 1.0
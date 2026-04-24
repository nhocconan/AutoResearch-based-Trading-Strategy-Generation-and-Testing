#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based stoploss.
- Uses 4h timeframe (primary) and 1d HTF for EMA50 trend alignment
- Donchian levels calculated from prior 4h OHLC: upper = max(high[20]), lower = min(low[20])
- Breakout logic: long when price closes above upper band with uptrend,
                  short when price closes below lower band with downtrend
- Trend filter: only long when 4h close > 1d EMA50, only short when 4h close < 1d EMA50
- Discrete signal size: 0.25 to balance reward and risk, minimizing fee churn
- Stoploss: exit when price reverses to opposite Donchian band (mean reversion)
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe as per research
- Works in both bull/bear: trend filter avoids counter-trend trades, Donchian breakouts capture momentum in all regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll_max
    donchian_lower = low_roll_min
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 1d EMA50 and 4h Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above upper band AND uptrend (close > EMA50)
            if close[i] > donchian_upper[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below lower band AND downtrend (close < EMA50)
            elif close[i] < donchian_lower[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below lower band (mean reversion)
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above upper band (mean reversion)
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0
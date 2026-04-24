#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 12h EMA trend filter and ATR volatility filter.
- Donchian(20) upper/lower bands from previous 20 periods (breakout on close)
- Long when price breaks above upper band AND price > 12h EMA50 (uptrend) AND ATR(14) > 0.3 * ATR(50)
- Short when price breaks below lower band AND price < 12h EMA50 (downtrend) AND ATR(14) > 0.3 * ATR(50)
- Exit when price reverts to opposite Donchian band OR volatility drops below threshold
- Designed to capture strong breakouts with trend alignment and volatility confirmation
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 75-200 total trades over 4 years (19-50/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h OHLC for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough data for EMA50
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Trend filter: price above/below 12h EMA50
    uptrend = close > ema_50_12h_aligned
    downtrend = close < ema_50_12h_aligned
    
    # ATR-based volatility filter: ATR(14) > 0.3 * ATR(50)
    # True Range calculation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    volatility_filter = atr_14 > (0.3 * atr_50)
    
    # Donchian channels (20-period) - using previous 20 periods for breakout
    # Upper band: highest high of previous 20 periods
    # Lower band: lowest low of previous 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 50, 20)  # Need 12h EMA50, ATR data, Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(volatility_filter[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian band AND uptrend AND sufficient volatility
            if close[i] > highest_high[i] and uptrend[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band AND downtrend AND sufficient volatility
            elif close[i] < lowest_low[i] and downtrend[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to lower Donchian band OR volatility drops
            if close[i] < lowest_low[i] or not volatility_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to upper Donchian band OR volatility drops
            if close[i] > highest_high[i] or not volatility_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_VolatilityFilter_v1"
timeframe = "4h"
leverage = 1.0
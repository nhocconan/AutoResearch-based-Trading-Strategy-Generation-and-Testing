#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and ATR-based volatility filter.
- Long when price breaks above Donchian(20) high AND price > 1d EMA34 (uptrend) AND ATR(14) > 0.3 * ATR(50)
- Short when price breaks below Donchian(20) low AND price < 1d EMA34 (downtrend) AND ATR(14) > 0.3 * ATR(50)
- Exit when price crosses the midpoint of Donchian(20) channel
- Designed to capture breakouts in trending markets with volatility confirmation
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 20-50 total trades over 4 years (5-12/year)
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
    
    # Donchian(20) channel
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    midpoint = (highest_high + lowest_low) / 2.0
    
    # Breakout signals
    breakout_up = close > highest_high
    breakout_down = close < lowest_low
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Trend filter
    uptrend = close > ema_34_1d_aligned
    downtrend = close < ema_34_1d_aligned
    
    # ATR-based volatility filter: ATR(14) > 0.3 * ATR(50)
    # True Range calculation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    volatility_filter = atr_14 > (0.3 * atr_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 34, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volatility_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above upper band AND uptrend AND sufficient volatility
            if breakout_up[i] and uptrend[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower band AND downtrend AND sufficient volatility
            elif breakout_down[i] and downtrend[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below midpoint
            if close[i] < midpoint[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above midpoint
            if close[i] > midpoint[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_VolatilityFilter_v1"
timeframe = "4h"
leverage = 1.0
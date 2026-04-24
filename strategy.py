#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA34 trend filter and ATR-based volatility filter.
- Uses 4h timeframe (primary) and 12h HTF for EMA34 trend alignment.
- Donchian channels calculated from prior 20 periods (4h) to avoid look-ahead.
- Breakout logic: long when price closes above upper band with ATR filter,
                  short when price closes below lower band with ATR filter.
- Trend filter: only long when 4h close > 12h EMA34, only short when 4h close < 12h EMA34.
- Volatility filter: ATR(14) > 0.5 * ATR(50) to avoid low-volatility choppy markets.
- Discrete signal size: 0.25 to balance reward and risk, minimizing fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in both bull/bear: trend filter avoids counter-trend trades, Donchian breakouts capture momentum in all regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate ATR(14) and ATR(50) for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr_14 > (0.5 * atr_50)
    
    # Calculate Donchian channels from prior 20 periods (4h)
    # Use rolling window on prior data to avoid look-ahead
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 20)  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volatility_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above upper Donchian band AND uptrend AND sufficient volatility
            if close[i] > donchian_upper[i] and uptrend[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below lower Donchian band AND downtrend AND sufficient volatility
            elif close[i] < donchian_lower[i] and downtrend[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to middle of Donchian channels or reverse signal
            donchian_mid = (donchian_upper[i] + donchian_lower[i]) / 2
            if close[i] <= donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to middle of Donchian channels or reverse signal
            donchian_mid = (donchian_upper[i] + donchian_lower[i]) / 2
            if close[i] >= donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA34_ATRVolFilter_v1"
timeframe = "4h"
leverage = 1.0
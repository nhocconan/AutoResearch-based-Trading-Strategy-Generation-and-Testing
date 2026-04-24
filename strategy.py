#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
- Uses 6h timeframe (primary) and 1d/1w HTF for multi-timeframe alignment
- Donchian levels calculated from prior 20-period 6h high/low: upper = max(high[-20:-1]), lower = min(low[-20:-1])
- Breakout logic: long when price closes above upper band with volume spike and weekly uptrend,
                  short when price closes below lower band with volume spike and weekly downtrend
- Trend filter: only long when 6h close > 1w EMA50, only short when 6h close < 1w EMA50
- Volume confirmation: current 6h volume > 1.8 * 20-period 6h volume MA to capture institutional interest
- Discrete signal size: 0.25 to balance reward and risk, minimizing fee churn
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe as per research
- Works in both bull/bear: trend filter avoids counter-trend trades, Donchian breakouts capture momentum in all regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 6h volume MA for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    # Calculate prior 20-period Donchian levels (shifted by 1 to avoid look-ahead)
    # Upper band = max of previous 20 highs (excluding current bar)
    # Lower band = min of previous 20 lows (excluding current bar)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 1w EMA50 for trend filter (using 1w HTF data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 21)  # Need 20-period Donchian and 1w EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above upper band AND weekly uptrend AND volume spike
            if close[i] > donchian_upper[i] and close[i] > ema_50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below lower band AND weekly downtrend AND volume spike
            elif close[i] < donchian_lower[i] and close[i] < ema_50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to midline (mean reversion) or reverse signal
            midline = (donchian_upper[i] + donchian_lower[i]) / 2
            if close[i] <= midline:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to midline (mean reversion) or reverse signal
            midline = (donchian_upper[i] + donchian_lower[i]) / 2
            if close[i] >= midline:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1wEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0
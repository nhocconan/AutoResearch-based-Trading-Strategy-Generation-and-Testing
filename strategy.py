#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 12h EMA trend filter and volume confirmation.
- Uses 4h timeframe (primary) and 12h HTF for EMA50 trend alignment
- Donchian levels calculated from prior 20 periods (4h): upper = max(high), lower = min(low)
- Breakout logic: long when price closes above upper band with volume spike and uptrend,
                  short when price closes below lower band with volume spike and downtrend
- Trend filter: only long when 4h EMA > 12h EMA50, only short when 4h EMA < 12h EMA50
- Volume confirmation: current 4h volume > 1.5 * 20-period 4h volume MA to capture institutional interest
- Discrete signal size: 0.25 to balance reward and risk, minimizing fee churn
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
    volume = prices['volume'].values
    
    # Calculate 4h EMA21 for trend confirmation (faster than EMA50)
    ema_21_4h = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate prior 20-period Donchian channels (4h)
    # Need to shift by 1 to avoid look-ahead (use prior completed period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    # Trend filter: 4h EMA21 vs 12h EMA50
    uptrend = ema_21_4h > ema_50_12h_aligned
    downtrend = ema_21_4h < ema_50_12h_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 21, 20)  # Need 12h EMA50, 4h EMA21, and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above upper band AND uptrend AND volume spike
            if close[i] > donchian_upper[i] and uptrend[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below lower band AND downtrend AND volume spike
            elif close[i] < donchian_lower[i] and downtrend[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to middle of channel (mean reversion) or reverse signal
            donchian_middle = (donchian_upper[i] + donchian_lower[i]) / 2
            if close[i] <= donchian_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to middle of channel (mean reversion) or reverse signal
            donchian_middle = (donchian_upper[i] + donchian_lower[i]) / 2
            if close[i] >= donchian_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0
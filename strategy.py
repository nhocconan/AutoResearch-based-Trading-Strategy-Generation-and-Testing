#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h HMA21 trend filter and volume spike confirmation.
- Uses 4h timeframe (primary) and 12h HTF for HMA21 trend alignment
- Donchian channels calculated from prior 20-period 4h high/low (no look-ahead)
- Breakout logic: long when price closes above upper band with volume spike and uptrend,
                  short when price closes below lower band with volume spike and downtrend
- Trend filter: only long when 12h HMA21 > prior 12h HMA21, only short when 12h HMA21 < prior 12h HMA21
- Volume confirmation: current 4h volume > 2.0 * 20-period 4h volume MA
- Discrete signal size: 0.25 to balance reward and risk, minimizing fee churn
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe as per research
- Works in both bull/bear: trend filter avoids counter-trend trades, Donchian breakouts capture momentum in all regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = pd.Series(series).ewm(span=half_period, adjust=False, min_periods=half_period).mean().values
    # WMA of full period
    wma_full = pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    # Final HMA: WMA of raw_hma with sqrt_period
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False, min_periods=sqrt_period).mean().values
    return hma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian channels (20-period, prior close to avoid look-ahead)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 12h HMA21 for trend confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    hma_21_12h = calculate_hma(close_12h, 21)
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # Trend filter: 12h HMA21 rising/falling (compare to prior value)
    hma_rising = hma_21_12h_aligned > np.roll(hma_21_12h_aligned, 1)
    hma_falling = hma_21_12h_aligned < np.roll(hma_21_12h_aligned, 1)
    # Handle first value
    hma_rising[0] = False
    hma_falling[0] = False
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(hma_21_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above upper band AND HMA rising AND volume spike
            if close[i] > donchian_upper[i] and hma_rising[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below lower band AND HMA falling AND volume spike
            elif close[i] < donchian_lower[i] and hma_falling[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to middle band (mean reversion) or reverse signal
            donchian_middle = (donchian_upper[i] + donchian_lower[i]) / 2
            if close[i] <= donchian_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to middle band (mean reversion) or reverse signal
            donchian_middle = (donchian_upper[i] + donchian_lower[i]) / 2
            if close[i] >= donchian_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hHMA21_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0
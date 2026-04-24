#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly EMA50 trend filter and volume spike confirmation.
- Uses 6h timeframe (primary) and 1w HTF for EMA50 trend alignment
- Donchian levels calculated from prior 20-period 6h high/low: upper = max(high[-20:-1]), lower = min(low[-20:-1])
- Breakout logic: long when price closes above upper Donchian with volume spike and weekly uptrend,
                  short when price closes below lower Donchian with volume spike and weekly downtrend
- Trend filter: only long when 6h close > 1w EMA50, only short when 6h close < 1w EMA50
- Volume confirmation: current 6h volume > 2.0 * 20-period 6h volume MA (stricter to reduce trades)
- Discrete signal size: 0.25 to balance reward and risk, minimizing fee churn
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe as per research
- Weekly trend filter avoids counter-trend trades in bear markets, Donchian breakouts capture momentum in all regimes
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
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate prior 20-period Donchian levels (shifted by 1 to avoid look-ahead)
    # Use rolling window on prior bars only: max(high[i-20:i-1]), min(low[i-20:i-1])
    high_shifted = np.roll(high, 1)
    low_shifted = np.roll(low, 1)
    high_shifted[0] = np.nan
    low_shifted[0] = np.nan
    
    donchian_upper = pd.Series(high_shifted).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_shifted).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Trend filter: 6h close vs 1w EMA50
    uptrend = close > ema_50_1w_aligned
    downtrend = close < ema_50_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 20)  # Need 1w EMA50, Donchian20, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above upper Donchian AND uptrend AND volume spike
            if close[i] > donchian_upper[i] and uptrend[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below lower Donchian AND downtrend AND volume spike
            elif close[i] < donchian_lower[i] and downtrend[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below lower Donchian (breakdown) or reverse signal
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above upper Donchian (breakout) or reverse signal
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1wEMA50_VolumeSpike_v2"
timeframe = "6h"
leverage = 1.0
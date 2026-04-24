#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Uses 1d timeframe (primary) and 1w HTF for trend alignment (proven pattern from DB)
- Donchian channel from previous 20 completed 1d bars (structure-based breakout)
- Long when price breaks above upper Donchian AND price > 1w EMA50 (uptrend) AND volume > 1.5 * volume MA(20)
- Short when price breaks below lower Donchian AND price < 1w EMA50 (downtrend) AND volume > 1.5 * volume MA(20)
- Exit when price reverts to the middle of the Donchian channel (mean reversion structure)
- Discrete signal size: 0.25 to minimize fee churn
- Target: 30-100 total trades over 4 years (7-25/year) as per 1d timeframe recommendation
- Works in both bull/bear: trend filter avoids counter-trend trades, Donchian breakouts capture momentum in all regimes
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
    
    # Calculate 1w EMA50 for trend filter (using previous completed 1w bar)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough data for EMA50
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian channel (20-period high/low)
    # Use rolling window on 1d data, then align to 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough data for Donchian
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper = max(high_1d over 20 periods)
    # Donchian lower = min(low_1d over 20 periods)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (previous 20-bar Donchian available at open)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    donchian_mid = (donchian_high_aligned + donchian_low_aligned) / 2.0  # Middle for exit
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * volume_ma)
    
    # Trend filter: price above/below 1w EMA50
    uptrend = close > ema_50_1w_aligned
    downtrend = close < ema_50_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 1w EMA50, Donchian(20), volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian AND uptrend AND volume confirmation
            if close[i] > donchian_high_aligned[i] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND downtrend AND volume confirmation
            elif close[i] < donchian_low_aligned[i] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to middle of Donchian channel
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to middle of Donchian channel
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0
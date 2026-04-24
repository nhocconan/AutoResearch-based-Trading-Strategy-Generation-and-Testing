#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Long when price breaks above 20-day high AND 1w close > 1w EMA50 (bullish regime)
- Short when price breaks below 20-day low AND 1w close < 1w EMA50 (bearish regime)
- Volume confirmation: current volume > 1.5 * 20-day average volume
- Exit on opposite Donchian level (exit long on 20-day low, exit short on 20-day high)
- Uses 1d primary with 1w HTF to target 30-100 total trades over 4 years (7-25/year)
- Donchian provides trend-following structure; EMA50 filters regime; volume confirms momentum
- Designed to work in bull markets (breakouts with trend) and bear markets (breakouts against trend)
- Signal size: 0.25 discrete levels to minimize fee churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Trend filter: bullish if close > EMA50, bearish if close < EMA50
    bullish_regime = close > ema_50_1w_aligned
    bearish_regime = close < ema_50_1w_aligned
    
    # Calculate 20-day Donchian channels using 1d HTF data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 20-day high and low
    donchian_high = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (already aligned, but ensure proper shifting)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20) + 1  # Need EMA50 and Donchian(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above 20-day high AND bullish regime AND volume confirmation
            if close[i] > donchian_high_aligned[i] and bullish_regime[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below 20-day low AND bearish regime AND volume confirmation
            elif close[i] < donchian_low_aligned[i] and bearish_regime[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below 20-day low (opposite level)
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above 20-day high (opposite level)
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot regime filter and volume confirmation.
- Long when price breaks above 6h Donchian upper (20) AND weekly close > weekly pivot (bullish regime)
- Short when price breaks below 6h Donchian lower (20) AND weekly close < weekly pivot (bearish regime)
- Volume confirmation: current volume > 1.5 * 20-period average volume
- Exit on opposite Donchian breakout (lower for long exit, upper for short exit)
- Uses 6h primary with 1w HTF to target 75-200 trades over 4 years (19-50/year)
- Weekly pivot provides major support/resistance; Donchian captures breakouts; volume confirms momentum
- Designed to work in both bull (breakouts with regime) and bear (mean reversion at extremes) markets
- Signal size: 0.25 discrete levels to minimize fee churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly OHLC for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Get weekly OHLC arrays
    weekly_open = df_1w['open'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate weekly pivot point: (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe (waits for completed weekly bar)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Weekly regime: bullish if weekly close > pivot, bearish if weekly close < pivot
    bullish_regime = weekly_close > weekly_pivot
    bearish_regime = weekly_close < weekly_pivot
    
    # Align regime to 6h timeframe
    bullish_regime_aligned = align_htf_to_ltf(prices, df_1w, bullish_regime.astype(float))
    bearish_regime_aligned = align_htf_to_ltf(prices, df_1w, bearish_regime.astype(float))
    
    # Calculate 6h Donchian channels (20-period)
    # Donchian upper = highest high of last 20 periods
    # Donchian lower = lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 20  # Need Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(bullish_regime_aligned[i]) or 
            np.isnan(bearish_regime_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian upper AND bullish weekly regime AND volume confirmation
            if close[i] > donchian_upper[i] and bullish_regime_aligned[i] > 0.5 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian lower AND bearish weekly regime AND volume confirmation
            elif close[i] < donchian_lower[i] and bearish_regime_aligned[i] > 0.5 and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below Donchian lower (opposite level)
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above Donchian upper (opposite level)
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1wPivot_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0
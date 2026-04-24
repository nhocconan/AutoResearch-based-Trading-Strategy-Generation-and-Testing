#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 12h trend filter and volume confirmation.
- Primary timeframe: 6h for lower trade frequency (target: 50-150 total trades over 4 years).
- HTF: 12h EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Donchian(20): Upper/lower bands from prior 20-period high/low on 6h chart.
- Volume: Current 6h volume > 1.5 * 20-period volume MA to capture institutional interest.
- Entry: Long when price breaks above Donchian upper band AND 12h EMA50 bullish AND volume spike.
         Short when price breaks below Donchian lower band AND 12h EMA50 bearish AND volume spike.
- Exit: Price reverts to mid-point of Donchian channel OR loss of volume confirmation.
- Signal size: 0.25 discrete to balance return and drawdown.
This strategy targets breakouts with institutional participation in the direction of the 12h trend,
avoiding counter-trend trades. Works in both bull and bear markets by only taking trades in the
direction of the 12h trend, with volume spikes confirming participation. Donchian breakouts provide
clear entry/exit levels, while the 12h EMA50 filter ensures we trade with the intermediate trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    df_12h_close = df_12h['close'].values
    ema_12h = pd.Series(df_12h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period 12h volume MA
    df_12h_volume = df_12h['volume'].values
    vol_ma_12h = pd.Series(df_12h_volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian(20) on 6h: upper band = 20-period high, lower band = 20-period low
    # We need to calculate this manually since we can't use future data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Align HTF indicators to 6h
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Volume confirmation: current 6h volume > 1.5 * 20-period 12h volume MA (aligned)
    volume_spike = volume > (1.5 * vol_ma_12h_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough bars for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(donchian_mid[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Check for breakout signals with volume spike
            if volume_spike[i]:
                # Bullish breakout: price > Donchian upper AND 12h EMA50 bullish (close > EMA)
                if curr_close > donchian_upper[i] and curr_close > ema_12h_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price < Donchian lower AND 12h EMA50 bearish (close < EMA)
                elif curr_close < donchian_lower[i] and curr_close < ema_12h_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to mid-point OR loss of volume confirmation
            if curr_close <= donchian_mid[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to mid-point OR loss of volume confirmation
            if curr_close >= donchian_mid[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0
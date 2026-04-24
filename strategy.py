#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 12h for lower trade frequency and better signal quality.
- HTF: 1w EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Volume: Current 12h volume > 2.0 * 20-period volume MA to capture institutional interest.
- Donchian: Upper/Lower channels calculated from 20-period high/low.
- Entry: Long when price breaks above upper channel AND 1w EMA50 bullish AND volume spike.
         Short when price breaks below lower channel AND 1w EMA50 bearish AND volume spike.
- Exit: Price reverts to 20-period midpoint or loss of volume confirmation.
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
This strategy combines institutional volume confirmation with Donchian breakouts,
filtered by weekly trend to avoid counter-trend trades. Works in both bull and bear markets
by only taking trades in the direction of the 1w trend, with volume spikes confirming
participation. Donchian channels provide natural support/resistance for mean reversion exits.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    df_1w_close = df_1w['close'].values
    ema_1w = pd.Series(df_1w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period 1w volume MA
    df_1w_volume = df_1w['volume'].values
    vol_ma_1w = pd.Series(df_1w_volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period) from 1w data
    df_1w_high = df_1w['high'].values
    df_1w_low = df_1w['low'].values
    
    # Upper channel = 20-period high
    donchian_upper = pd.Series(df_1w_high).rolling(window=20, min_periods=20).max().values
    # Lower channel = 20-period low
    donchian_lower = pd.Series(df_1w_low).rolling(window=20, min_periods=20).min().values
    # Middle channel = (upper + lower) / 2
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Align HTF indicators to 12h
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1w, donchian_middle)
    
    # Volume confirmation: current 12h volume > 2.0 * 20-period 1w volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1w_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough bars for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(donchian_middle_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for breakout signals with volume spike
            if volume_spike[i]:
                # Bullish breakout: price > upper channel AND 1w EMA50 bullish (close > EMA)
                if curr_close > donchian_upper_aligned[i] and curr_close > ema_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price < lower channel AND 1w EMA50 bearish (close < EMA)
                elif curr_close < donchian_lower_aligned[i] and curr_close < ema_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to middle channel OR loss of volume confirmation
            if curr_close <= donchian_middle_aligned[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to middle channel OR loss of volume confirmation
            if curr_close >= donchian_middle_aligned[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0
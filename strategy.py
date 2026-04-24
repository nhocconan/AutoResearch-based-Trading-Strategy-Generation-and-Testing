#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 12h for lower trade frequency and reduced fee drag.
- HTF: 1d EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Volume: Current 12h volume > 1.8 * 20-period volume MA to capture institutional interest.
- Donchian: Upper/lower bands from 20-period high/low on 12h timeframe.
- Entry: Long when price breaks above upper band AND 1d EMA50 bullish AND volume spike.
         Short when price breaks below lower band AND 1d EMA50 bearish AND volume spike.
- Exit: Price reverts to 20-period midpoint (mean of bands) or loss of volume confirmation.
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
This strategy captures medium-term breakouts filtered by daily trend, with volume spikes
confirming institutional participation. Works in both bull and bear markets by only taking
trades in the direction of the 1d trend, avoiding counter-trend whipsaws.
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period 1d volume MA
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough bars for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Calculate 20-period Donchian bands on 12h data (using available history)
        if i >= 20:
            period_high = np.max(high[i-19:i+1])  # 20-period high including current
            period_low = np.min(low[i-19:i+1])   # 20-period low including current
            donchian_upper = period_high
            donchian_lower = period_low
            donchian_mid = (donchian_upper + donchian_lower) / 2
        else:
            # Not enough data for bands
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.8 * 20-period 1d volume MA (aligned)
        volume_spike = curr_volume > (1.8 * vol_ma_1d_aligned[i])
        
        if position == 0:
            # Check for breakout signals with volume spike
            if volume_spike:
                # Bullish breakout: price > upper band AND 1d EMA50 bullish (close > EMA)
                if curr_close > donchian_upper and curr_close > ema_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price < lower band AND 1d EMA50 bearish (close < EMA)
                elif curr_close < donchian_lower and curr_close < ema_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to midpoint OR loss of volume confirmation
            if curr_close <= donchian_mid or not volume_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to midpoint OR loss of volume confirmation
            if curr_close >= donchian_mid or not volume_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA50_Trend_VolumeConfirmation_v1"
timeframe = "12h"
leverage = 1.0
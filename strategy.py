#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 4h to limit trade frequency and reduce fee drag.
- HTF: 12h EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Volume: Current 4h volume > 2.0 * 20-period volume MA to capture institutional interest.
- Donchian: Upper/lower bands from 20-period high/low on 4h chart.
- Entry: Long when price breaks above upper band AND 12h EMA50 bullish AND volume spike.
         Short when price breaks below lower band AND 12h EMA50 bearish AND volume spike.
- Exit: Price reverts to 20-period EMA midpoint or loss of volume confirmation.
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 100-200 total trades over 4 years (25-50/year) for 4h timeframe.
This strategy captures breakouts aligned with the 12h trend, confirmed by volume spikes,
avoiding counter-trend trades. Works in both bull and bear markets by only taking trades
in the direction of the 12h trend, with volume confirming participation. Donchian bands
provide objective breakout levels, and EMA midpoint offers a logical mean-reversion exit.
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
    
    # Calculate 20-period Donchian channels on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2  # Midpoint for exit
    
    # Align HTF indicators to 4h
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Volume confirmation: current 4h volume > 2.0 * 20-period 12h volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_12h_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 20)  # Need enough bars for EMA50, volume MA, and Donchian
    
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
                # Bullish breakout: price > upper band AND 12h EMA50 bullish (close > EMA)
                if curr_close > donchian_upper[i] and curr_close > ema_12h_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price < lower band AND 12h EMA50 bearish (close < EMA)
                elif curr_close < donchian_lower[i] and curr_close < ema_12h_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to midpoint OR loss of volume confirmation
            if curr_close <= donchian_mid[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to midpoint OR loss of volume confirmation
            if curr_close >= donchian_mid[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 1d EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 4h to limit trade frequency and reduce fee drag.
- HTF: 1d EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Volume: Current 4h volume > 2.0 * 20-period volume MA to capture institutional participation.
- Donchian: Upper/lower bands from 20-period high/low on 4h chart.
- Entry: Long when price breaks above upper band AND 1d EMA50 bullish AND volume spike.
         Short when price breaks below lower band AND 1d EMA50 bearish AND volume spike.
- Exit: Price reverts to 20-period midline (mean of upper/lower bands) or loss of volume confirmation.
- Signal size: 0.25 discrete to minimize fee churn.
- Target: 100-200 total trades over 4 years (25-50/year) for 4h timeframe.
This strategy combines breakout momentum with trend and volume filters to avoid false breakouts.
Works in both bull and bear markets by only taking trades in the direction of the 1d trend,
with volume spikes confirming institutional participation. Donchian exit provides systematic
mean reversion to secure profits.
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
    
    # Get 1d data for EMA50 trend filter and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period 1d volume MA
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 20-period Donchian channels on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2  # Midline for exit
    
    # Align HTF indicators to 4h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 4h volume > 2.0 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough bars for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(donchian_mid[i])):
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
                # Bullish breakout: price > upper band AND 1d EMA50 bullish (close > EMA)
                if curr_close > donchian_upper[i] and curr_close > ema_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price < lower band AND 1d EMA50 bearish (close < EMA)
                elif curr_close < donchian_lower[i] and curr_close < ema_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to midline OR loss of volume confirmation
            if curr_close <= donchian_mid[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to midline OR loss of volume confirmation
            if curr_close >= donchian_mid[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0
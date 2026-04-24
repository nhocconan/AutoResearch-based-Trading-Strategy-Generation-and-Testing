#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 1d for lower trade frequency (target: 30-100 total trades over 4 years).
- HTF: 1w EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Volume: Current 1d volume > 2.0 * 20-period volume MA to capture institutional interest.
- Donchian: Upper/lower bands calculated from prior 20 periods (using prior day's data to avoid look-ahead).
- Entry: Long when price breaks above upper band AND 1w EMA50 bullish AND volume spike.
         Short when price breaks below lower band AND 1w EMA50 bearish AND volume spike.
- Exit: Price reverts to midpoint of Donchian channel or loss of volume confirmation.
- Signal size: 0.25 discrete to balance return and drawdown.
This strategy captures breakouts in the direction of the weekly trend with volume confirmation,
avoiding counter-trend trades. Works in both bull and bear markets by only taking trades
in the direction of the 1w trend, with volume spikes confirming participation.
Donchian channels provide objective breakout levels, and exits on mean reversion to the
channel midpoint reduce giving back profits in ranging markets.
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
    
    # Get 1d data for Donchian bands and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    df_1w_close = df_1w['close'].values
    ema_1w = pd.Series(df_1w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period 1d volume MA
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian bands from prior 20 periods (using shift(1) to avoid look-ahead)
    # Upper band = max(high, lookback=20)
    # Lower band = min(low, lookback=20)
    # Midpoint = (upper + lower) / 2
    h1d = df_1d['high'].values
    l1d = df_1d['low'].values
    
    # Rolling max/min on 1d data with lookback=20, then shift by 1 to use prior period only
    upper_band_raw = pd.Series(h1d).rolling(window=20, min_periods=20).max().values
    lower_band_raw = pd.Series(l1d).rolling(window=20, min_periods=20).min().values
    upper_band = np.roll(upper_band_raw, 1)  # Shift to use prior period's max
    lower_band = np.roll(lower_band_raw, 1)  # Shift to use prior period's min
    # Set first value to NaN since no prior period exists
    upper_band[0] = np.nan
    lower_band[0] = np.nan
    midpoint = (upper_band + lower_band) / 2
    
    # Align HTF indicators to 1d
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    midpoint_aligned = align_htf_to_ltf(prices, df_1d, midpoint)
    
    # Volume confirmation: current 1d volume > 2.0 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready (need at least 1 for roll shift + 20 for lookback + 50 for EMA)
    start_idx = max(50, 21)  # 1w EMA50 needs 50, Donchian needs 20+1=21
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(midpoint_aligned[i])):
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
                # Bullish breakout: price > upper band AND 1w EMA50 bullish (close > EMA)
                if curr_close > upper_band_aligned[i] and curr_close > ema_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price < lower band AND 1w EMA50 bearish (close < EMA)
                elif curr_close < lower_band_aligned[i] and curr_close < ema_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to midpoint OR loss of volume confirmation
            if curr_close <= midpoint_aligned[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to midpoint OR loss of volume confirmation
            if curr_close >= midpoint_aligned[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0
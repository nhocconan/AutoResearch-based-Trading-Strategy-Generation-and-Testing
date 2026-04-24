#!/usr/bin/env python3
"""
Hypothesis: 1h Donchian breakout with 1d trend filter and volume spike confirmation.
- Primary timeframe: 1h for precise entry/exit timing.
- HTF: 1d Donchian(20) for trend direction (bullish if close > upper band, bearish if close < lower band).
- Volume: Current 1h volume > 2.0 * 20-period volume MA to capture institutional interest.
- Entry: Long when price breaks above 1h Donchian(20) upper band AND 1d trend bullish AND volume spike.
         Short when price breaks below 1h Donchian(20) lower band AND 1d trend bearish AND volume spike.
- Exit: Opposite Donchian breakout (price < lower band for long, price > upper band for short) or loss of volume confirmation.
- Signal size: 0.20 discrete to limit drawdown and reduce fee churn.
- Session filter: Only trade between 08:00-20:00 UTC to avoid low-liquidity periods.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
This strategy uses Donchian channels for breakout confirmation in the direction of the higher timeframe trend,
with volume spikes to filter for institutional participation. Works in both bull and bear markets by
only taking trades aligned with the 1d trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1h Donchian(20)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Get 1d data for Donchian trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian(20)
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    df_1d_volume = df_1d['volume'].values
    
    highest_high_1d = pd.Series(df_1d_high).rolling(window=20, min_periods=20).max().values
    lowest_low_1d = pd.Series(df_1d_low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period 1d volume MA
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 1h
    highest_high_1d_aligned = align_htf_to_ltf(prices, df_1d, highest_high_1d)
    lowest_low_1d_aligned = align_htf_to_ltf(prices, df_1d, lowest_low_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 1h volume > 2.0 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20)  # Need enough bars for Donchian channels
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(highest_high_1d_aligned[i]) or np.isnan(lowest_low_1d_aligned[i]) or 
            np.isnan(volume_spike[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals with volume spike and session filter
            if volume_spike[i]:
                # Bullish: price breaks above 1h Donchian upper band AND 1d trend bullish (close > upper band)
                if curr_high > highest_high[i] and curr_close > highest_high_1d_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                # Bearish: price breaks below 1h Donchian lower band AND 1d trend bearish (close < lower band)
                elif curr_low < lowest_low[i] and curr_close < lowest_low_1d_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price breaks below 1h Donchian lower band OR loss of volume confirmation OR outside session
            if curr_low < lowest_low[i] or not volume_spike[i] or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above 1h Donchian upper band OR loss of volume confirmation OR outside session
            if curr_high > highest_high[i] or not volume_spike[i] or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_DonchianBreakout_1dDonchianTrend_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0
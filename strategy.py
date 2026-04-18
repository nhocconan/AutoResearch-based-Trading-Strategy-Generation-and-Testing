#!/usr/bin/env python3
"""
1d_WeeklyTrend_DonchianBreakout_VolumeFilter
Hypothesis: Trade Donchian(20) breakouts on daily timeframe with weekly trend filter and volume confirmation. Long when price breaks above 20-day high, weekly close > weekly open (bullish weekly candle), and volume > 1.5x 20-day average volume. Short when price breaks below 20-day low, weekly close < weekly open (bearish weekly candle), and volume > 1.5x average. Uses weekly trend to avoid counter-trend trades in strong weekly trends. Volume confirmation ensures breakouts have institutional interest. Designed for low trade frequency (<25/year) to minimize fee impact while capturing major moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly trend: bullish if close > open, bearish if close < open
    weekly_bullish = df_1w['close'].values > df_1w['open'].values
    weekly_bearish = df_1w['close'].values < df_1w['open'].values
    
    # Align weekly trend to daily
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Donchian channels (20-day)
    lookback = 20
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    if len(high) >= lookback:
        for i in range(lookback, len(high)):
            highest_high[i] = np.max(high[i-lookback:i])
            lowest_low[i] = np.min(low[i-lookback:i])
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, vol_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(weekly_bullish_aligned[i]) or 
            np.isnan(weekly_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above 20-day high, weekly bullish, volume confirmation
            if close[i] > highest_high[i] and weekly_bullish_aligned[i] > 0.5 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low, weekly bearish, volume confirmation
            elif close[i] < lowest_low[i] and weekly_bearish_aligned[i] > 0.5 and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 20-day low or weekly turns bearish
            if close[i] < lowest_low[i] or weekly_bearish_aligned[i] > 0.5:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 20-day high or weekly turns bullish
            if close[i] > highest_high[i] or weekly_bullish_aligned[i] > 0.5:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyTrend_DonchianBreakout_VolumeFilter"
timeframe = "1d"
leverage = 1.0
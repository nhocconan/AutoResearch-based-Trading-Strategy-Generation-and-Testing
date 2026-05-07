#!/usr/bin/env python3
# 1D_RangeBreakout_WeeklyTrend_VolumeFilter
# Hypothesis: On daily timeframe, take long when price breaks above weekly Donchian high
# in an up-trending market (weekly close > weekly EMA20) with volume confirmation.
# Take short when price breaks below weekly Donchian low in a down-trending market
# (weekly close < weekly EMA20) with volume confirmation. Uses weekly trend filter
# to avoid counter-trend trades. Designed for low trade frequency (~10-25/year) to
# minimize fee drag and work in both bull and bear markets.

name = "1D_RangeBreakout_WeeklyTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly close for EMA calculation
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Weekly EMA20 for trend filter
    ema20_1w = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Weekly Donchian channels (20-period)
    # For Donchian high: max of last 20 weekly highs
    # For Donchian low: min of last 20 weekly lows
    dh_series = pd.Series(weekly_high).rolling(window=20, min_periods=20).max()
    dl_series = pd.Series(weekly_low).rolling(window=20, min_periods=20).min()
    donchian_high = dh_series.values
    donchian_low = dl_series.values
    
    # Align weekly indicators to daily timeframe
    ema20_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Volume filter: current volume > 1.5x average volume (50-period)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure we have volume MA and weekly indicators
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema20_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high + Uptrend (weekly close > weekly EMA20) + volume
            if (close[i] > donchian_high_aligned[i] and 
                weekly_close[-1] > ema20_1w[-1] if len(weekly_close) > 0 else False and  # Current weekly close > weekly EMA
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian low + Downtrend (weekly close < weekly EMA20) + volume
            elif (close[i] < donchian_low_aligned[i] and 
                  weekly_close[-1] < ema20_1w[-1] if len(weekly_close) > 0 else False and  # Current weekly close < weekly EMA
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price falls back below weekly Donchian high or trend turns down
            if close[i] < donchian_high_aligned[i] or weekly_close[-1] < ema20_1w[-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises back above weekly Donchian low or trend turns up
            if close[i] > donchian_low_aligned[i] or weekly_close[-1] > ema20_1w[-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
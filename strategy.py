#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with weekly volume confirmation and 1w trend filter.
# Uses weekly Donchian channels to establish long-term trend direction.
# Enters long when price breaks above 12h Donchian upper (20) and weekly trend is up.
# Enters short when price breaks below 12h Donchian lower (20) and weekly trend is down.
# Volume confirmation requires current 12h volume > 1.5x 20-period weekly average volume.
# Designed for low trade frequency (10-30/year) to minimize fee drag on 12h timeframe.
# Works in both bull and bear markets by following the weekly trend.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for trend filter and volume average
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    donchian_high_20_w = pd.Series(high_w).rolling(window=20, min_periods=20).max().values
    donchian_low_20_w = pd.Series(low_w).rolling(window=20, min_periods=20).min().values
    
    # Weekly trend: price above/below midpoint of weekly Donchian
    donchian_mid_w = (donchian_high_20_w + donchian_low_20_w) / 2
    close_w = df_1w['close'].values
    weekly_uptrend = close_w > donchian_mid_w
    weekly_downtrend = close_w < donchian_mid_w
    
    # Weekly volume average for confirmation
    vol_w = df_1w['volume'].values
    vol_ma_20_w = pd.Series(vol_w).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly indicators to 12h
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    vol_ma_20_w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_w)
    
    # Calculate 12h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align weekly volume average to 12h
    vol_ma_20_w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_w)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]) or
            np.isnan(vol_ma_20_w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        close = prices['close'].iloc[i]
        vol_current = prices['volume'].iloc[i]
        
        # Trend filters
        is_uptrend = weekly_uptrend_aligned[i] > 0.5
        is_downtrend = weekly_downtrend_aligned[i] > 0.5
        
        # Volume confirmation: current 12h volume > 1.5x 20-week average volume
        volume_confirm = vol_current > 1.5 * vol_ma_20_w_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above 12h Donchian high + weekly uptrend + volume
            if close > donchian_high_20[i] and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 12h Donchian low + weekly downtrend + volume
            elif close < donchian_low_20[i] and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below 12h Donchian low OR weekly trend turns down
                if close < donchian_low_20[i]:
                    exit_signal = True
                elif not is_uptrend:
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above 12h Donchian high OR weekly trend turns up
                if close > donchian_high_20[i]:
                    exit_signal = True
                elif not is_downtrend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0
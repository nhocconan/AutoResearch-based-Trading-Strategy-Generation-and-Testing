#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with 1-week Donchian breakout and volume confirmation.
# Uses weekly Donchian channels for trend direction and breakout signals.
# Enters on weekly channel breakouts in the direction of the trend, confirmed by volume.
# Exits on opposite Donchian breakout or volume divergence.
# Designed for 20-30 trades/year on 1d timeframe to minimize fee drag.
# Weekly trend filter reduces whipsaw in sideways markets and improves win rate.

name = "1d_1w_donchian_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for weekly calculations
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need enough for Donchian(20)
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly Donchian upper and lower bands (20-period)
    donch_high_20 = np.full_like(high_1w, np.nan, dtype=float)
    donch_low_20 = np.full_like(low_1w, np.nan, dtype=float)
    
    for i in range(19, len(high_1w)):
        donch_high_20[i] = np.max(high_1w[i-19:i+1])
        donch_low_20[i] = np.min(low_1w[i-19:i+1])
    
    # Weekly trend: price above mid-channel = bullish, below = bearish
    mid_channel = (donch_high_20 + donch_low_20) / 2
    weekly_trend_bull = close_1w > mid_channel
    weekly_trend_bear = close_1w < mid_channel
    
    # Weekly volume average (20-period) for confirmation
    vol_avg_20 = np.full_like(volume_1w, np.nan, dtype=float)
    for i in range(19, len(volume_1w)):
        vol_avg_20[i] = np.mean(volume_1w[i-19:i+1])
    
    # Align weekly indicators to daily timeframe
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1w, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1w, donch_low_20)
    weekly_trend_bull_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_bull)
    weekly_trend_bear_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_bear)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after enough data for calculations
        # Skip if any required data is invalid
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i]) or
            np.isnan(weekly_trend_bull_aligned[i]) or np.isnan(weekly_trend_bear_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current weekly volume > 1.5 * 20-period average
        # Note: We use the aligned weekly volume average
        vol_filter = volume_1w[min(i // 7, len(volume_1w)-1)] > 1.5 * vol_avg_20_aligned[i] if not np.isnan(vol_avg_20_aligned[i]) else False
        
        # Determine weekly trend direction
        is_bullish_week = weekly_trend_bull_aligned[i]
        is_bearish_week = weekly_trend_bear_aligned[i]
        
        # Breakout signals: price breaks above/below weekly Donchian channels
        breakout_long = high[i] >= donch_high_20_aligned[i] and vol_filter
        breakout_short = low[i] <= donch_low_20_aligned[i] and vol_filter
        
        # Exit signals: opposite breakout or trend reversal
        exit_long = (position == 1 and 
                    (low[i] <= donch_low_20_aligned[i] or  # Opposite Donchian break
                     not is_bullish_week))  # Trend reversal
        exit_short = (position == -1 and 
                      (high[i] >= donch_high_20_aligned[i] or  # Opposite Donchian break
                       not is_bearish_week))  # Trend reversal
        
        # Entry logic: breakout in direction of weekly trend
        if breakout_long and is_bullish_week and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_short and is_bearish_week and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
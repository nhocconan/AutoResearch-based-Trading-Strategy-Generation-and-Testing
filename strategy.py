#!/usr/bin/env python3
# 1d_WeeklyDonchian_Breakout_With_Volume_Trend_Filter
# Hypothesis: Weekly Donchian breakouts capture major trends, with daily volume and trend filters to reduce false signals.
# Weekly trend filter (price above/below weekly EMA) ensures we trade with the higher timeframe momentum.
# Volume confirmation ensures institutional participation. Designed for low trade frequency to minimize fee drag.
# Target: 15-25 trades/year (60-100 total over 4 years) on BTC/ETH.

name = "1d_WeeklyDonchian_Breakout_With_Volume_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels and EMA trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    donchian_high = np.full_like(weekly_high, np.nan)
    donchian_low = np.full_like(weekly_low, np.nan)
    
    for i in range(len(weekly_high)):
        if i >= 19:  # 20-period lookback
            donchian_high[i] = np.max(weekly_high[i-19:i+1])
            donchian_low[i] = np.min(weekly_low[i-19:i+1])
    
    # Align weekly Donchian levels to daily
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # Calculate weekly EMA (21-period) for trend filter
    weekly_close = df_weekly['close'].values
    ema_21 = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_weekly, ema_21)
    
    # Daily volume confirmation: volume > 1.5x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 21)  # Ensure Donchian and EMA are calculated
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_21_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high + price above weekly EMA + volume confirmation
            if close[i] > donchian_high_aligned[i] and close[i] > ema_21_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low + price below weekly EMA + volume confirmation
            elif close[i] < donchian_low_aligned[i] and close[i] < ema_21_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below weekly Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above weekly Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
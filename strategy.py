#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d 20-day Donchian breakout with 1-week EMA200 trend filter and volume confirmation
# This strategy trades breakouts of the 20-day price channel, aligned with weekly trend (EMA200)
# and confirmed by volume spikes. Works in bull markets (long breakouts) and bear markets 
# (short breakouts) by following the weekly trend direction. Uses discrete position sizing (0.25)
# to balance return and minimize transaction costs. Target: 15-25 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Donchian channels and EMA trend (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-day Donchian channels for previous day
    # Upper = max(high of last 20 days)
    # Lower = min(low of last 20 days)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (same timeframe, no shift needed)
    donchian_upper_aligned = donchian_upper  # Already at 1d frequency
    donchian_lower_aligned = donchian_lower  # Already at 1d frequency
    
    # 1-week EMA(200) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to ensure Donchian data is ready
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close breaks above Donchian upper + above weekly EMA200 + volume spike
            if close[i] > donchian_upper_aligned[i] and close[i] > ema_200_1w_aligned[i] and volume[i] > 2.0 * vol_avg_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian lower + below weekly EMA200 + volume spike
            elif close[i] < donchian_lower_aligned[i] and close[i] < ema_200_1w_aligned[i] and volume[i] > 2.0 * vol_avg_20[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses weekly EMA200 in opposite direction
            if position == 1:
                # Exit long: Close below weekly EMA200
                if close[i] < ema_200_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Close above weekly EMA200
                if close[i] > ema_200_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_WeeklyEMA200_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0
#!/usr/bin/env python3
# Hypothesis: 1h timeframe with 4h Donchian channel breakout and 1d EMA200 trend filter.
# Uses 4h Donchian breakout for entry signals and 1d EMA200 for trend filter.
# Trend filter ensures trades only in direction of higher timeframe trend to reduce whipsaw.
# Session filter (08-20 UTC) reduces noise trades. Position size fixed at 0.20.
# Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag.

name = "1h_Donchian_Breakout_4hTrend_1dEMA200_Session"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channel (20-period high/low)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian upper/lower (20-period)
    donchian_high = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian levels to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA200 trend filter
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Trend conditions
    trend_up = close > ema_200_1d_aligned
    trend_down = close < ema_200_1d_aligned
    
    # Breakout conditions: price must close beyond Donchian levels
    breakout_up = close > donchian_high_aligned
    breakout_down = close < donchian_low_aligned
    
    # Volume filter: current volume > 1.5x 24-period average volume
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(trend_up[i]) or np.isnan(trend_down[i]) or
            np.isnan(volume_filter[i]) or np.isnan(session_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above 4h Donchian high + 1d uptrend + volume + session
            if breakout_up[i] and trend_up[i] and volume_filter[i] and session_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: breakout below 4h Donchian low + 1d downtrend + volume + session
            elif breakout_down[i] and trend_down[i] and volume_filter[i] and session_filter[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price returns to 4h Donchian low or trend reversal
            if close[i] <= donchian_low_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price returns to 4h Donchian high or trend reversal
            if close[i] >= donchian_high_aligned[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
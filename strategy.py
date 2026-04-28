#!/usr/bin/env python3
# Hypothesis: 1d Donchian breakout with weekly EMA trend filter and volume confirmation.
# Donchian(20) breakouts on daily timeframe capture significant trends.
# Weekly EMA(50) filters for long-term trend alignment to avoid counter-trend trades.
# Volume spike (2x 20-day average) confirms breakout validity.
# Works in bull markets (catching uptrends via upper band breaks) and bear markets (catching downtrends via lower band breaks).
# Targets 10-30 total trades over 4 years (3-8/year) with position sizing 0.25 to minimize fee drag.

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
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels from daily data (need daily high/low)
    # Since we're on 1d timeframe, we can use the prices directly
    donchian_len = 20
    upper = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    lower = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    
    # Volume filter: volume > 2x 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, donchian_len, 20)  # Wait for EMA, Donchian, and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA(50)
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Donchian breakout conditions
        breakout_upper = high[i] > upper[i-1]  # Break above upper band
        breakdown_lower = low[i] < lower[i-1]  # Break below lower band
        
        # Entry conditions with volume spike confirmation
        long_entry = uptrend and breakout_upper and volume_spike[i]
        short_entry = downtrend and breakdown_lower and volume_spike[i]
        
        # Exit conditions: trend reversal or opposite Donchian break
        long_exit = (not uptrend) or breakdown_lower
        short_exit = (not downtrend) or breakout_upper
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Donchian_20_WeeklyEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: Daily Donchian Breakout with Weekly Trend Filter
# Hypothesis: Donchian channel breakouts capture major trend moves, confirmed by weekly trend direction.
# Works in bull markets via upper band breakouts, in bear via lower band breakdowns.
# Weekly trend filter prevents counter-trend trades, reducing false signals.
# Target: 15-25 trades/year to minimize fee drag.
name = "daily_donchian20_weekly_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    close_weekly = df_weekly['close'].values
    ema_weekly = pd.Series(close_weekly).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Calculate daily Donchian channels (20 period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_weekly_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend: price above/below weekly EMA50
        weekly_uptrend = close[i] > ema_weekly_aligned[i]
        weekly_downtrend = close[i] < ema_weekly_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower (trend reversal)
            if close[i] < donchian_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper (trend reversal)
            if close[i] > donchian_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price breaks above Donchian upper + weekly uptrend
            if close[i] > donchian_upper[i] and weekly_uptrend:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian lower + weekly downtrend
            elif close[i] < donchian_lower[i] and weekly_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d/1w strategy using weekly Donchian breakout with daily trend filter
# In bull markets: buy weekly highs above weekly EMA, sell on weekly lows
# In bear markets: sell weekly lows below weekly EMA, buy on weekly highs
# Weekly timeframe reduces noise and whipsaw, daily trend filter adds confirmation
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag
# Works in both regimes by following weekly momentum with daily trend confirmation

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for HTF calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 20-week Donchian channels
    high_20w = np.full(len(close_1w), np.nan)
    low_20w = np.full(len(close_1w), np.nan)
    for i in range(20, len(close_1w)):
        high_20w[i] = np.max(high_1w[i-20:i])
        low_20w[i] = np.min(low_1w[i-20:i])
    
    # Calculate 50-week EMA for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_50w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to daily timeframe
    high_20w_aligned = align_htf_to_ltf(prices, df_1w, high_20w)
    low_20w_aligned = align_htf_to_ltf(prices, df_1w, low_20w)
    ema_50w_aligned = align_htf_to_ltf(prices, df_1w, ema_50w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(60, n):  # Start after sufficient data
        # Skip if data not ready
        if (np.isnan(high_20w_aligned[i]) or 
            np.isnan(low_20w_aligned[i]) or 
            np.isnan(ema_50w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Daily trend filter: price above/below weekly EMA
        above_weekly_ema = close[i] > ema_50w_aligned[i]
        below_weekly_ema = close[i] < ema_50w_aligned[i]
        
        # Weekly Donchian breakout conditions
        weekly_high_break = close[i] > high_20w_aligned[i]
        weekly_low_break = close[i] < low_20w_aligned[i]
        
        # Entry: breakout in direction of weekly trend
        long_entry = weekly_high_break and above_weekly_ema
        short_entry = weekly_low_break and below_weekly_ema
        
        # Exit: opposite breakout or trend reversal
        exit_long = position == 1 and (weekly_low_break or below_weekly_ema)
        exit_short = position == -1 and (weekly_high_break or above_weekly_ema)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_donchian_ema50_breakout"
timeframe = "1d"
leverage = 1.0
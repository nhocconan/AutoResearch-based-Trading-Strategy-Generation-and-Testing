#!/usr/bin/env python3
"""
1d_1w_Trend_Filtered_Breakout_v1
Hypothesis: Daily timeframe with weekly trend filter and breakout of prior day's high/low.
Trades only in direction of weekly trend (using weekly EMA) to avoid counter-trend losses.
Uses volume confirmation to filter false breakouts. Designed for low trade frequency
(~10-25 trades/year) to minimize fee drag while capturing trending moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Trend_Filtered_Breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA (21 period) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Daily high/low of previous day for breakout levels
    # Note: We use the previous day's high/low to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = np.nan  # First bar has no previous day
    prev_low[0] = np.nan
    
    # Volume average (20 period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(prev_high[i]) or np.isnan(prev_low[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume spike: current volume > 1.5x average
        volume_spike = volume[i] > vol_ma[i] * 1.5
        
        # Trend filter: price above/below weekly EMA
        above_weekly_ema = close[i] > ema_1w_aligned[i]
        below_weekly_ema = close[i] < ema_1w_aligned[i]
        
        # Breakout conditions: price breaks previous day's high/low
        breakout_high = close[i] > prev_high[i]
        breakout_low = close[i] < prev_low[i]
        
        # Entry conditions: breakout in direction of weekly trend with volume
        long_entry = breakout_high and volume_spike and above_weekly_ema
        short_entry = breakout_low and volume_spike and below_weekly_ema
        
        # Exit conditions: price returns to previous day's close or trend reversal
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        long_exit = close[i] < prev_close[i] if not np.isnan(prev_close[i]) else False
        short_exit = close[i] > prev_close[i] if not np.isnan(prev_close[i]) else False
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
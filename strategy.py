#!/usr/bin/env python3
"""
12h_Alligator_Mouth_Close_With_WeeklyTrend
Hypothesis: Williams Alligator (13,8,5 SMAs) mouth closes when lines converge in ranging markets.
Trade when price breaks out of closed mouth in direction of weekly EMA50 trend.
Uses volume confirmation to avoid false breakouts. Works in bull/bear via trend filter.
Target: 20-40 trades/year to minimize fee drag while capturing directional moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_weekly = df_weekly['close'].values
    ema_50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    # Calculate Williams Alligator on 12h data
    # Jaw (13-period SMMA), Teeth (8-period), Lips (5-period)
    # SMMA is smoothed moving average: similar to EMA but with alpha = 1/period
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (Prev SMMA * (period-1) + Close) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Mouth closed when max-min of three lines is small relative to ATR
    # But simpler: mouth closes when all three lines are within 0.5% of each other
    max_teeth = np.maximum.reduce([jaw, teeth, lips])
    min_teeth = np.minimum.reduce([jaw, teeth, lips])
    mouth_width = max_teeth - min_teeth
    # Normalize by price to get percentage
    mouth_width_pct = mouth_width / close * 100
    
    # Mouth closed when width < 0.3%
    mouth_closed = mouth_width_pct < 0.3
    
    # Breakout when price closes outside the Alligator lines
    # Long breakout: close > max(Jaw, Teeth, Lips)
    # Short breakout: close < min(Jaw, Teeth, Lips)
    breakout_up = close > max_teeth
    breakout_down = close < min_teeth
    
    # Volume confirmation: >1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_weekly_aligned[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(mouth_closed[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: weekly EMA50 direction
        uptrend = close[i] > ema_50_weekly_aligned[i]
        downtrend = close[i] < ema_50_weekly_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Entry: breakout of closed mouth in trend direction with volume
        long_entry = vol_confirm and uptrend and breakout_up[i] and mouth_closed[i]
        short_entry = vol_confirm and downtrend and breakout_down[i] and mouth_closed[i]
        
        # Exit: price returns inside Alligator mouth or trend change
        inside_mouth = (close[i] >= min_teeth[i]) and (close[i] <= max_teeth[i])
        long_exit = inside_mouth or (not uptrend)
        short_exit = inside_mouth or (not downtrend)
        
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
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Alligator_Mouth_Close_With_WeeklyTrend"
timeframe = "12h"
leverage = 1.0
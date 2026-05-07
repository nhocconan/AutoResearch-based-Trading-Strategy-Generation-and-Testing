#!/usr/bin/env python3
"""
1d_WedgeBreakout_1wTrend_v1
Hypothesis: On daily timeframe, detect ascending/descending wedge patterns combined with weekly trend filter.
Enter long when price breaks above descending wedge resistance with weekly uptrend.
Enter short when price breaks below ascending wedge support with weekly downtrend.
Wedges provide high-probability breakout signals, weekly trend filters reduce counter-trend trades.
Designed for low frequency (target 10-25 trades/year) to minimize fee impact in 2025 bear market.
"""
name = "1d_WedgeBreakout_1wTrend_v1"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate wedge pattern: converging trendlines
    # Descending wedge: lower highs + higher lows (bullish)
    # Ascending wedge: higher highs + lower lows (bearish)
    
    # Find swing points (simplified: local peaks/troughs over 5 periods)
    def find_swing_highs(arr, window=5):
        highs = np.full_like(arr, np.nan)
        for i in range(window, len(arr) - window):
            if arr[i] == np.max(arr[i-window:i+window+1]):
                highs[i] = arr[i]
        return highs
    
    def find_swing_lows(arr, window=5):
        lows = np.full_like(arr, np.nan)
        for i in range(window, len(arr) - window):
            if arr[i] == np.min(arr[i-window:i+window+1]):
                lows[i] = arr[i]
        return lows
    
    # Get recent swing points for trendline calculation
    swing_highs = find_swing_highs(high, 3)
    swing_lows = find_swing_lows(low, 3)
    
    # Calculate trendlines using linear regression on last 3 swing points
    def calc_trendline(points, lookback=20):
        valid = ~np.isnan(points)
        if np.sum(valid) < 3:
            return np.full_like(points, np.nan)
        
        # Get indices and values of valid points
        indices = np.where(valid)[0]
        values = points[valid]
        
        # Use last 3 points for trendline
        if len(indices) >= 3:
            idx_last3 = indices[-3:]
            val_last3 = values[-3:]
            # Linear fit: y = mx + b
            A = np.vstack([idx_last3, np.ones(len(idx_last3))]).T
            m, b = np.linalg.lstsq(A, val_last3, rcond=None)[0]
            # Generate trendline values
            trendline = m * np.arange(len(points)) + b
            return trendline
        return np.full_like(points, np.nan)
    
    # Resistance trendline (from swing highs)
    resistance = calc_trendline(swing_highs, 20)
    # Support trendline (from swing lows)
    support = calc_trendline(swing_lows, 20)
    
    # Wedge conditions
    # Descending wedge: resistance sloping down, support sloping up
    # Ascending wedge: resistance sloping up, support sloping down
    
    # Calculate slopes of trendlines (last 5 periods)
    def calc_slope(arr, lookback=5):
        slopes = np.full_like(arr, np.nan)
        for i in range(lookback, len(arr)):
            if not np.isnan(arr[i]) and not np.isnan(arr[i-lookback]):
                slopes[i] = (arr[i] - arr[i-lookback]) / lookback
        return slopes
    
    resist_slope = calc_slope(resistance, 5)
    support_slope = calc_slope(support, 5)
    
    # Wedge detection
    descending_wedge = (resist_slope < 0) & (support_slope > 0)  # Falling resistance, rising support
    ascending_wedge = (resist_slope > 0) & (support_slope < 0)   # Rising resistance, falling support
    
    # Breakout conditions
    # Long: price breaks above resistance in descending wedge
    # Short: price breaks below support in ascending wedge
    
    # Volume confirmation: volume > 1.5 * 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(resist_slope[i]) or 
            np.isnan(support_slope[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: descending wedge breakout + weekly uptrend + volume
            if (descending_wedge[i] and 
                close[i] > resistance[i] and 
                close[i] > ema_50_1w_aligned[i] and  # Price above weekly EMA50 (uptrend)
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: ascending wedge breakdown + weekly downtrend + volume
            elif (ascending_wedge[i] and 
                  close[i] < support[i] and 
                  close[i] < ema_50_1w_aligned[i] and  # Price below weekly EMA50 (downtrend)
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit conditions: opposite wedge breakout or trend reversal
            if position == 1:  # Long position
                # Exit: price breaks below support OR weekly trend turns down
                if (close[i] < support[i] or 
                    close[i] < ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                # Exit: price breaks above resistance OR weekly trend turns up
                if (close[i] > resistance[i] or 
                    close[i] > ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
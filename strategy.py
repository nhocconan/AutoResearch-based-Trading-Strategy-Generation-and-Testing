#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Weekly Donchian Breakout with Volume Filter
# Hypothesis: Donchian(20) breakouts on 12h timeframe in direction of weekly trend
# with volume confirmation capture sustained moves while avoiding whipsaws.
# Weekly trend filter ensures alignment with higher timeframe momentum.
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.

name = "12h_weekly_donchian_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and breakout levels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA(21) for trend direction
    weekly_close = df_weekly['close'].values
    ema_21 = pd.Series(weekly_close).ewm(span=21, adjust=False).mean().values
    
    # Weekly Donchian channels (20-period high/low)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    high_series = pd.Series(weekly_high)
    low_series = pd.Series(weekly_low)
    weekly_high_20 = high_series.rolling(window=20, min_periods=20).max().values
    weekly_low_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align weekly indicators to 12h timeframe
    ema_21_aligned = align_htf_to_ltf(prices, df_weekly, ema_21)
    high_20_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low_20)
    
    # Volume filter on 12h: volume > 1.3x 30-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_21_aligned[i]) or np.isnan(high_20_aligned[i]) or
            np.isnan(low_20_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls back below weekly low or trend changes
            if close[i] < low_20_aligned[i] or close[i] < ema_21_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises back above weekly high or trend changes
            if close[i] > high_20_aligned[i] or close[i] > ema_21_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Trend filter: price above/below weekly EMA
            if close[i] > ema_21_aligned[i]:
                # Long entry: breakout above weekly high with volume
                if (high[i] > high_20_aligned[i] and close[i] > high_20_aligned[i] and
                    vol_filter[i]):
                    position = 1
                    signals[i] = 0.25
            elif close[i] < ema_21_aligned[i]:
                # Short entry: breakdown below weekly low with volume
                if (low[i] < low_20_aligned[i] and close[i] < low_20_aligned[i] and
                    vol_filter[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals
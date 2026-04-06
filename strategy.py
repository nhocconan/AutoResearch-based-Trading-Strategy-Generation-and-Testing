#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly pivot direction filter and volume confirmation.
# Uses 1d OHLC to calculate weekly pivot points (based on prior week), aligned to 6h bars.
# Long when price breaks above Donchian(20) high AND above weekly pivot (bullish bias).
# Short when price breaks below Donchian(20) low AND below weekly pivot (bearish bias).
# Weekly pivot provides structural bias; Donchian breakout provides entry timing.
# Volume filter ensures breakouts have participation.
# Works in bull (breakouts with trend) and bear (breakdowns with trend) markets.
# Target: 60-120 total trades over 4 years (15-30/year).

name = "6h_donchian20_weeklypivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for weekly pivot calculation (using prior week's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:  # need at least a week of data
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week's OHLC
    # Group by week (Monday start) and get prior week's OHLC
    df_1d = df_1d.copy()
    df_1d['week'] = pd.to_datetime(df_1d.index).isocalendar().week
    df_1d['year'] = pd.to_datetime(df_1d.index).isocalendar().year
    
    # Get weekly OHLC
    weekly = df_1d.groupby(['year', 'week']).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    if len(weekly) < 2:
        return np.zeros(n)
    
    # Prior week's OHLC for pivot calculation
    prior_weekly = weekly.iloc[:-1]  # exclude current (incomplete) week
    if len(prior_weekly) == 0:
        return np.zeros(n)
    
    # Calculate pivot points for each day using prior week's OHLC
    # We'll use the most recent completed week's OHLC for all days in current week
    week_high = prior_weekly['high'].values[-1]   # highest high of prior week
    week_low = prior_weekly['low'].values[-1]     # lowest low of prior week
    week_close = prior_weekly['close'].values[-1] # close of prior week
    
    # Standard pivot point formula
    pivot = (week_high + week_low + week_close) / 3
    # Support/resistance levels
    r1 = 2 * pivot - week_low
    s1 = 2 * pivot - week_high
    r2 = pivot + (week_high - week_low)
    s2 = pivot - (week_high - week_low)
    # We'll use S1 as bullish bias threshold, R1 as bearish bias threshold
    bullish_bias = s1   # above this = bullish bias
    bearish_bias = r1   # below this = bearish bias
    
    # Create arrays aligned to daily index
    bullish_bias_arr = np.full(len(df_1d), bullish_bias)
    bearish_bias_arr = np.full(len(df_1d), bearish_bias)
    
    # Align to 6h bars
    bullish_bias_aligned = align_htf_to_ltf(prices, df_1d, bullish_bias_arr)
    bearish_bias_aligned = align_htf_to_ltf(prices, df_1d, bearish_bias_arr)
    
    # Donchian channels (20-period) on 6h data
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume moving average for filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(bullish_bias_aligned[i]) or np.isnan(bearish_bias_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: require volume above average
        vol_filter = volume[i] > vol_ma[i]
        
        if position == 1:  # long position
            # Exit: price breaks below Donchian low or loses bullish bias
            if close[i] < donchian_low[i] or close[i] < bullish_bias_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high or loses bearish bias
            if close[i] > donchian_high[i] or close[i] > bearish_bias_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume filter
            if vol_filter:
                # Long: break above Donchian high AND above bullish bias
                if close[i] > donchian_high[i] and close[i] > bullish_bias_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: break below Donchian low AND below bearish bias
                elif close[i] < donchian_low[i] and close[i] < bearish_bias_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Daily Donchian Breakout with Volume and ADX Filter
# Hypothesis: 12h price breaking out of 20-period Donchian channels with
# volume confirmation and ADX > 25 works in both bull and bear markets.
# In bull markets: buy breakouts above upper channel.
# In bear markets: sell breakdowns below lower channel.
# Target: 15-30 trades/year (60-120 over 4 years) - within 50-150 total limit.

name = "12h_donchian20_volume_adx_v1"
timeframe = "12h"
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
    
    # Get daily data for Donchian channels
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on daily data
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    
    # Upper and lower channels (20-period high/low)
    upper_channel = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    upper_channel = np.roll(upper_channel, 1)
    lower_channel = np.roll(lower_channel, 1)
    
    # Handle first element
    if len(upper_channel) > 1:
        upper_channel[0] = upper_channel[1]
        lower_channel[0] = lower_channel[1]
    else:
        upper_channel[0] = 0
        lower_channel[0] = 0
    
    # Align to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_daily, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_daily, lower_channel)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # ADX filter: ADX > 25 indicates trending market
    # Calculate ADX components
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        else:
            plus_dm[i] = 0
            
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        else:
            minus_dm[i] = 0
            
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth the values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    adx_filter = adx > 25
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below lower channel or ADX weakens
            if close[i] < lower_aligned[i] or not adx_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price crosses above upper channel or ADX weakens
            if close[i] > upper_aligned[i] or not adx_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: breakout above upper channel with volume and ADX
            if close[i] > upper_aligned[i] and vol_filter[i] and adx_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below lower channel with volume and ADX
            elif close[i] < lower_aligned[i] and vol_filter[i] and adx_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
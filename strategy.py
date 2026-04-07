#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Daily Donchian Breakout with Volume and ADX Trend Filter
# Hypothesis: Price breaking Donchian channels (20-period) with volume confirmation and ADX trend filter works in both bull and bear markets.
# In bull markets: buy breakouts above upper band. In bear markets: sell breakdowns below lower band.
# Uses daily Donchian for structure and 4h for execution, targeting 20-50 trades/year.

name = "4h_daily_donchian_breakout_volume_adx_v1"
timeframe = "4h"
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
    
    # Get daily data for Donchian calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    
    # Upper and lower bands
    upper = np.full_like(daily_high, np.nan)
    lower = np.full_like(daily_low, np.nan)
    
    for i in range(20, len(daily_high)):
        upper[i] = np.max(daily_high[i-20:i])
        lower[i] = np.min(daily_low[i-20:i])
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    upper = np.roll(upper, 1)
    lower = np.roll(lower, 1)
    
    # Handle first element
    if len(upper) > 1:
        upper[0] = upper[1]
        lower[0] = lower[1]
    else:
        upper[0] = 0
        lower[0] = 0
    
    # Align to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_daily, upper)
    lower_aligned = align_htf_to_ltf(prices, df_daily, lower)
    
    # ADX filter: trend strength (14-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # True Range
    tr1 = high_series - low_series
    tr2 = np.abs(high_series - close_series.shift(1))
    tr3 = np.abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_series - high_series.shift(1)
    down_move = low_series.shift(1) - low_series
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    plus_di = 100 * (pd.Series(plus_dm).ewm(span=14, min_periods=14).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(span=14, min_periods=14).mean() / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=14, min_periods=14).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in trending markets (ADX > 25)
        if adx[i] <= 25:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price below lower band or volume filter fails
            if close[i] < lower_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price above upper band or volume filter fails
            if close[i] > upper_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: breakout above upper band with volume
            if close[i] > upper_aligned[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below lower band with volume
            elif close[i] < lower_aligned[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
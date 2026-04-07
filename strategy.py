#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Daily Donchian Breakout with Volume and ADX Filter
# Hypothesis: Donchian(20) breakouts on 12h timeframe capture medium-term trends.
# Volume confirms institutional interest, ADX filters for trending markets.
# Works in bull markets (long breakouts) and bear markets (short breakdowns).
# Daily timeframe provides robust structure, 12h balances frequency and noise.
# Target: 12-37 trades/year (50-150 total over 4 years).

name = "12h_daily_donchian_breakout_volume_adx_v1"
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
    
    # Get daily data for Donchian channels
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Daily high/low
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    
    # Calculate Donchian channels (20-period high/low)
    high_series = pd.Series(daily_high)
    low_series = pd.Series(daily_low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian levels to 12h timeframe (shifted by 1 for completed bars)
    donchian_high_aligned = align_htf_to_ltf(prices, df_daily, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_daily, donchian_low)
    
    # Volume filter: volume > 1.5x 50-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=50, min_periods=50).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # ADX filter (14-period) for trending markets
    # Calculate +DM, -DM, TR
    high_series_12h = pd.Series(high)
    low_series_12h = pd.Series(low)
    close_series_12h = pd.Series(close)
    
    up_move = high_series_12h.diff()
    down_move = low_series_12h.diff()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr1 = high_series_12h - low_series_12h
    tr2 = np.abs(high_series_12h - close_series_12h.shift(1))
    tr3 = np.abs(low_series_12h - close_series_12h.shift(1))
    tr = pd.Series(np.maximum(tr1, np.maximum(tr2, tr3)))
    
    # Smoothed values
    atr = tr.ewm(alpha=1/14, adjust=False).mean().values
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_filter = adx > 25  # Trending market threshold
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls back below Donchian low (trailing stop)
            if close[i] < donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises back above Donchian high (trailing stop)
            if close[i] > donchian_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: breakout above Donchian high with volume and ADX
            if (high[i] > donchian_high_aligned[i] and close[i] > donchian_high_aligned[i] and
                vol_filter[i] and adx_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below Donchian low with volume and ADX
            elif (low[i] < donchian_low_aligned[i] and close[i] < donchian_low_aligned[i] and
                  vol_filter[i] and adx_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
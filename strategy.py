#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout with Volume and ADX Filter
# Hypothesis: Donchian channel breakouts capture strong trends. Volume confirms institutional participation.
# ADX > 25 filters for trending markets, avoiding whipsaws in ranging conditions.
# Works in both bull and bear markets: breakouts in direction of trend are followed, while false breakouts are filtered.
# Target: 15-35 trades/year (60-140 over 4 years).

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
    
    # Get daily data for ADX calculation (trend strength filter)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate ADX on daily data
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # True Range
    tr1 = np.abs(daily_high[1:] - daily_low[1:])
    tr2 = np.abs(daily_high[1:] - daily_close[:-1])
    tr3 = np.abs(daily_low[1:] - daily_close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with original index
    
    # Directional Movement
    dm_plus = np.where((daily_high[1:] - daily_high[:-1]) > (daily_low[:-1] - daily_low[1:]), 
                       np.maximum(daily_high[1:] - daily_high[:-1], 0), 0)
    dm_minus = np.where((daily_low[:-1] - daily_low[1:]) > (daily_high[1:] - daily_high[:-1]), 
                        np.maximum(daily_low[:-1] - daily_low[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[1:period])  # Skip index 0 (nan)
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, 14)
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    
    # Calculate Donchian channels (20-period) on 12h data
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume filter: volume > 1.3x 30-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls to Donchian low or ADX weakens
            if low[i] <= lowest_low[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises to Donchian high or ADX weakens
            if high[i] >= highest_high[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high with volume and strong trend
            if high[i] > highest_high[i] and vol_filter[i] and adx_aligned[i] > 25:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low with volume and strong trend
            elif low[i] < lowest_low[i] and vol_filter[i] and adx_aligned[i] > 25:
                position = -1
                signals[i] = -0.25
    
    return signals
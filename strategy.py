#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout + Volume + ADX Trend Filter
# Hypothesis: Donchian channel breakouts capture momentum. Volume confirms institutional participation.
# ADX filter ensures trades only in trending regimes (ADX > 25) to avoid whipsaws in ranging markets.
# Works in both bull and bear markets by taking breakouts in direction of trend.
# 12h timeframe targets 12-37 trades/year (50-150 over 4 years) to minimize fee drag.
name = "12h_donchian20_volume_adx_v2"
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
    
    # Get weekly data for trend filter (ADX calculation)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate ADX on weekly data
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(weekly_high[1:] - weekly_low[1:])
    tr2 = np.abs(weekly_high[1:] - weekly_close[:-1])
    tr3 = np.abs(weekly_low[1:] - weekly_close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # Directional Movement
    dm_plus = np.where((weekly_high[1:] - weekly_high[:-1]) > (weekly_low[:-1] - weekly_low[1:]), 
                       np.maximum(weekly_high[1:] - weekly_high[:-1], 0), 0)
    dm_minus = np.where((weekly_low[:-1] - weekly_low[1:]) > (weekly_high[1:] - weekly_high[:-1]), 
                        np.maximum(weekly_low[:-1] - weekly_low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/14)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[:period]) / period
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        alpha = 1.0 / period
        for i in range(period, len(data)):
            if not np.isnan(data[i]):
                result[i] = result[i-1] * (1 - alpha) + data[i] * alpha
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align weekly ADX to 12h
    adx_12h = align_htf_to_ltf(prices, df_1w, adx)
    
    # Donchian Channel (20-period) on 12h data
    lookback = 20
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(lookback-1, len(high)):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_12h[i] > 25
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower OR trend ends
            if close[i] < lowest_low[i] or not trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper OR trend ends
            if close[i] > highest_high[i] or not trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation and trending market
            if vol_filter[i] and trending:
                # Long: price breaks above Donchian upper
                if close[i] > highest_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian lower
                elif close[i] < lowest_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
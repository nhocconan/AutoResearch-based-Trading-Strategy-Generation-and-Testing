#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Mean Reversion with 1d ADX Trend Filter
# Williams %R identifies overbought/oversold conditions for mean reversion entries.
# 1d ADX > 25 filters for trending markets (avoid chop) while < 20 identifies ranging markets.
# In trending markets (ADX > 25): fade extreme %R readings (>80 or <20) in direction of trend.
# In ranging markets (ADX < 20): mean revert at %R extremes regardless of trend.
# Works in bull markets (buy oversold in uptrend) and bear markets (sell overbought in downtrend).
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
name = "6h_WilliamsR_MeanReversion_1dADXFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smooth TR, DM+ and DM- using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[1:period])
        # Subsequent values: smoothed = previous_smoothed - (previous_smoothed/period) + current
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    period = 14
    tr_smooth = wilder_smooth(tr, period)
    dm_plus_smooth = wilder_smooth(dm_plus, period)
    dm_minus_smooth = wilder_smooth(dm_minus, period)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = np.full_like(dx, np.nan)
    # First ADX value is average of first 'period' DX values
    if len(dx) >= 2*period:
        adx[2*period-1] = np.nanmean(dx[period:2*period])
        # Subsequent ADX values: Wilder smoothing of DX
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Williams %R on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r[i]
        adx_val = adx_aligned[i]
        close_val = close[i]
        
        if position == 0:
            # Entry conditions
            if adx_val > 25:  # Trending market
                # In uptrend: buy when oversold (%R < -80)
                # In downtrend: sell when overbought (%R > -20)
                # Determine trend direction from price vs 50-period SMA
                sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values[i]
                if not np.isnan(sma_50):
                    if close_val > sma_50 and wr < -80:  # Uptrend + oversold
                        signals[i] = 0.25
                        position = 1
                    elif close_val < sma_50 and wr > -20:  # Downtrend + overbought
                        signals[i] = -0.25
                        position = -1
            else:  # Ranging market (ADX < 25)
                # Mean revert at extreme %R levels
                if wr < -80:  # Oversold
                    signals[i] = 0.25
                    position = 1
                elif wr > -20:  # Overbought
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: %R reaches overbought (> -20) or centerline (-50)
            if wr > -20 or wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: %R reaches oversold (< -80) or centerline (-50)
            if wr < -80 or wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
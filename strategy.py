#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Weekly Donchian Breakout with Volume and ADX Filter
# Hypothesis: Donchian(20) breakouts on daily timeframe in direction of weekly ADX > 25
# trend with volume confirmation capture major trend moves while avoiding whipsaws.
# Weekly trend filter ensures alignment with major market direction, reducing false
# breakouts in chop. Designed for low trade frequency (10-25/year) to minimize fee drag.
# Works in bull markets via breakout momentum and in bear via trend-following shorts.

name = "1d_weekly_donchian_breakout_volume_adx_v1"
timeframe = "1d"
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
    
    # Get weekly data for ADX and breakout levels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Calculate ADX on weekly data
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # True Range
    tr1 = weekly_high[1:] - weekly_low[1:]
    tr2 = np.abs(weekly_high[1:] - weekly_close[:-1])
    tr3 = np.abs(weekly_low[1:] - weekly_close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((weekly_high[1:] - weekly_high[:-1]) > (weekly_low[:-1] - weekly_low[1:]),
                       np.maximum(weekly_high[1:] - weekly_high[:-1], 0), 0)
    dm_minus = np.where((weekly_low[:-1] - weekly_low[1:]) > (weekly_high[1:] - weekly_high[:-1]),
                        np.maximum(weekly_low[:-1] - weekly_low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[1:period]) / period
        # Subsequent values
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Weekly breakout levels (20-period high/low)
    high_series = pd.Series(weekly_high)
    low_series = pd.Series(weekly_low)
    weekly_high_20 = high_series.rolling(window=20, min_periods=20).max().values
    weekly_low_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align weekly indicators to daily timeframe
    adx_aligned = align_htf_to_ltf(prices, df_weekly, adx)
    high_20_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low_20)
    
    # Volume filter on daily: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(adx_aligned[i]) or np.isnan(high_20_aligned[i]) or
            np.isnan(low_20_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls back below 20-week low or ADX weakens
            if close[i] < low_20_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises back above 20-week high or ADX weakens
            if close[i] > high_20_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Strong trend required
            if adx_aligned[i] >= 25:
                # Long entry: breakout above 20-week high with volume
                if (high[i] > high_20_aligned[i] and close[i] > high_20_aligned[i] and
                    vol_filter[i]):
                    position = 1
                    signals[i] = 0.25
                # Short entry: breakdown below 20-week low with volume
                elif (low[i] < low_20_aligned[i] and close[i] < low_20_aligned[i] and
                      vol_filter[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals
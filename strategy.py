#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Weekly Donchian Breakout with Volume and ADX Filter
# Hypothesis: Weekly Donchian(20) breakouts with volume confirmation and daily ADX > 25
# filter capture strong momentum moves. Weekly timeframe provides robust trend filtering
# across bull/bear markets, reducing whipsaws. Target: 20-40 trades/year.

name = "4h_weekly_donchian_breakout_volume_adx_v1"
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
    
    # Get weekly data for Donchian levels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Get daily data for ADX filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate ADX on daily data
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # True Range
    tr1 = daily_high[1:] - daily_low[1:]
    tr2 = np.abs(daily_high[1:] - daily_close[:-1])
    tr3 = np.abs(daily_low[1:] - daily_close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((daily_high[1:] - daily_high[:-1]) > (daily_low[:-1] - daily_low[1:]),
                       np.maximum(daily_high[1:] - daily_high[:-1], 0), 0)
    dm_minus = np.where((daily_low[:-1] - daily_low[1:]) > (daily_high[1:] - daily_high[:-1]),
                        np.maximum(daily_low[:-1] - daily_low[1:], 0), 0)
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
    
    # Weekly Donchian levels (20-period high/low)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    high_series = pd.Series(weekly_high)
    low_series = pd.Series(weekly_low)
    weekly_high_20 = high_series.rolling(window=20, min_periods=20).max().values
    weekly_low_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align weekly and daily indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    weekly_high_20_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high_20)
    weekly_low_20_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low_20)
    
    # Volume filter on 4h: volume > 1.5x 30-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(adx_aligned[i]) or np.isnan(weekly_high_20_aligned[i]) or
            np.isnan(weekly_low_20_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls back below weekly low or ADX weakens
            if close[i] < weekly_low_20_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises back above weekly high or ADX weakens
            if close[i] > weekly_high_20_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Strong trend required
            if adx_aligned[i] >= 25:
                # Long entry: breakout above weekly high with volume
                if (high[i] > weekly_high_20_aligned[i] and close[i] > weekly_high_20_aligned[i] and
                    vol_filter[i]):
                    position = 1
                    signals[i] = 0.25
                # Short entry: breakdown below weekly low with volume
                elif (low[i] < weekly_low_20_aligned[i] and close[i] < weekly_low_20_aligned[i] and
                      vol_filter[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: Daily Donchian Breakout with Volume and ADX Filter
# Hypothesis: On the daily timeframe, Donchian(20) breakouts in the direction of 
# weekly ADX > 25 trend with volume confirmation capture momentum moves while 
# avoiding whipsaws. Weekly trend filter ensures robustness across bull/bear markets, 
# volume filter reduces false breakouts. Target: 15-25 trades/year (60-100 total) 
# to minimize fee drag.

name = "daily_donchian_breakout_volume_adx_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for ADX (trend filter)
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
    
    # Wilder's smoothing function
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
    
    # Daily breakout levels (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    daily_high_20 = high_series.rolling(window=20, min_periods=20).max().values
    daily_low_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align weekly ADX to daily timeframe
    adx_aligned = align_htf_to_ltf(prices, df_weekly, adx)
    
    # Volume filter on daily: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if required data not available
        if (np.isnan(adx_aligned[i]) or np.isnan(daily_high_20[i]) or
            np.isnan(daily_low_20[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls back below 20-day low or ADX weakens
            if close[i] < daily_low_20[i] or adx_aligned[i] < 25:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises back above 20-day high or ADX weakens
            if close[i] > daily_high_20[i] or adx_aligned[i] < 25:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Strong trend required
            if adx_aligned[i] >= 25:
                # Long entry: breakout above 20-day high with volume
                if (high[i] > daily_high_20[i] and close[i] > daily_high_20[i] and
                    vol_filter[i]):
                    position = 1
                    signals[i] = 0.25
                # Short entry: breakdown below 20-day low with volume
                elif (low[i] < daily_low_20[i] and close[i] < daily_low_20[i] and
                      vol_filter[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals
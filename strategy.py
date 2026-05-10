#150399
# 6h_Donchian_Breakout_WeeklyPivot_Trend
# Hypothesis: Combine Donchian breakouts with weekly pivot levels and trend filter to capture strong moves in both bull and bear markets.
# - Use 1d Donchian (20) for entry signals (breakout of weekly range)
# - Use 12h EMA50 for trend filter (align with medium-term momentum)
# - Use 1d weekly pivot (calculated from prior week) for directional bias (only trade long above weekly pivot, short below)
# - Require volume confirmation (volume > 1.5x 20-period MA) to avoid false breakouts
# - Target: 15-35 trades/year to minimize fee drag on 6h timeframe
# Works in bull markets (buy breakouts above weekly pivot in uptrend) and bear markets (sell breakdowns below weekly pivot in downtrend).

name = "6h_Donchian_Breakout_WeeklyPivot_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and weekly pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:  # Need ~20 for Donchian + extra for weekly calc
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate weekly pivot from prior week (using daily data)
    # Need to calculate weekly OHLC from daily data
    # We'll resample daily to weekly using pandas (only once, outside loop)
    df_1d_indexed = df_1d.copy()
    df_1d_indexed.index = pd.to_datetime(df_1d_indexed['open_time'])
    weekly = df_1d_indexed.resample('W').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    if len(weekly) < 2:
        return np.zeros(n)
    
    # Weekly pivot points (standard formula)
    weekly_high = weekly['high'].values
    weekly_low = weekly['low'].values
    weekly_close = weekly['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    # Align weekly pivot to 6h timeframe (weekly data needs alignment)
    # We need to align the weekly values to the 1d index first, then to 6h
    # Create a series aligned to daily index first
    weekly_pivot_daily = pd.Series(weekly_pivot, index=weekly.index)
    # Resample to daily (forward fill) to get daily aligned values
    weekly_pivot_daily_aligned = weekly_pivot_daily.reindex(df_1d_indexed.index, method='ffill').values
    # Now align from daily to 6h
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot_daily_aligned)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation (20-period MA on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian (20), weekly pivot (need weekly data), EMA50 (50), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: uptrend + price breaks above Donchian high + above weekly pivot + volume
            if uptrend and close[i] > donchian_high_aligned[i] and close[i] > weekly_pivot_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price breaks below Donchian low + below weekly pivot + volume
            elif downtrend and close[i] < donchian_low_aligned[i] and close[i] < weekly_pivot_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price breaks below weekly pivot
            if not uptrend or close[i] < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price breaks above weekly pivot
            if not downtrend or close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
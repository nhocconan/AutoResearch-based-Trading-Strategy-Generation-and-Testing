# 2025-06-21: 6h Donchian breakout + weekly pivot direction + volume confirmation
# Hypothesis: 6h Donchian(20) breakouts in direction of weekly pivot trend with volume confirmation.
# Weekly pivot acts as long-term trend filter. Volume confirms breakout strength.
# Works in bull/bear by using pivot direction (not price level) as trend filter.
# Target: 12-37 trades/year (50-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot calculation - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (based on previous weekly bar)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly_prev = df_weekly['close'].values
    
    # Standard pivot point: P = (H + L + C) / 3
    pp = (high_weekly + low_weekly + close_weekly_prev) / 3.0
    # Weekly pivot trend: above PP = bullish, below PP = bearish
    pp_trend = pp  # we'll use the actual PP value for comparison
    
    # Align weekly pivot to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp_trend)
    
    # Load daily data for Donchian calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) channels from daily data
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    
    # Upper channel: 20-day high
    upper_20 = pd.Series(high_daily).rolling(window=20, min_periods=20).max().values
    # Lower channel: 20-day low
    lower_20 = pd.Series(low_daily).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 6h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_daily, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_daily, lower_20)
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian(20) AND price > weekly pivot (bullish trend) with volume
            if (close[i] > upper_20_aligned[i] and 
                close[i] > pp_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian(20) AND price < weekly pivot (bearish trend) with volume
            elif (close[i] < lower_20_aligned[i] and 
                  close[i] < pp_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to the opposite Donchian channel or weekly pivot
            if position == 1:
                if close[i] < lower_20_aligned[i] or close[i] < pp_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > upper_20_aligned[i] or close[i] > pp_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_Donchian20_WeeklyPivot_Trend_Volume_Session"
timeframe = "6h"
leverage = 1.0
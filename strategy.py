#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Choppiness Index + Donchian Breakout with Volume Filter
# Uses weekly trend filter to avoid counter-trend trades. In trending markets (ADX > 25 on weekly),
# trade breakouts of daily Donchian channels only when choppiness is low (< 38.2) indicating trending regime.
# In ranging markets (Choppiness > 61.8), fade extremes at Bollinger Bands.
# Works in both bull and bear markets by following the weekly trend direction.
# Target: 20-60 trades per year.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Load daily data for Donchian and Bollinger calculations
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Calculate weekly ADX (14-period) for trend strength
    # True Range
    tr1 = weekly_high - weekly_low
    tr2 = np.abs(weekly_high - np.roll(weekly_close, 1))
    tr3 = np.abs(weekly_low - np.roll(weekly_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((weekly_high - np.roll(weekly_high, 1)) > (np.roll(weekly_low, 1) - weekly_low), 
                       np.maximum(weekly_high - np.roll(weekly_high, 1), 0), 0)
    dm_minus = np.where((np.roll(weekly_low, 1) - weekly_low) > (weekly_high - np.roll(weekly_high, 1)), 
                        np.maximum(np.roll(weekly_low, 1) - weekly_low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr_weekly = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr_weekly + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr_weekly + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_weekly = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly ADX to daily timeframe
    adx_weekly_aligned = align_htf_to_ltf(prices, df_weekly, adx_weekly)
    
    # Calculate daily Donchian channels (20-period)
    donchian_high = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to daily timeframe (already aligned since we used daily data)
    
    # Calculate daily Bollinger Bands (20, 2.0)
    bb_middle = pd.Series(daily_close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(daily_close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    
    # Calculate daily Choppiness Index (14-period)
    # True Range for each period
    tr1_daily = daily_high - daily_low
    tr2_daily = np.abs(daily_high - np.roll(daily_close, 1))
    tr3_daily = np.abs(daily_low - np.roll(daily_close, 1))
    tr_daily = np.maximum(tr1_daily, np.maximum(tr2_daily, tr3_daily))
    tr_daily[0] = tr1_daily[0]
    
    # Sum of true range over 14 periods
    atr_sum = pd.Series(tr_daily).rolling(window=14, min_periods=14).sum().values
    
    # Maximum high - minimum low over 14 periods
    max_high = pd.Series(daily_high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(daily_low).rolling(window=14, min_periods=14).min().values
    range_max = max_high - min_low
    
    # Choppiness Index = 100 * log10(atr_sum / range_max) / log10(14)
    # Avoid division by zero and log of zero
    ratio = np.where(range_max > 0, atr_sum / range_max, 1.0)
    chop = 100 * np.log10(np.maximum(ratio, 1e-10)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):  # Start after enough data for indicators
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(adx_weekly_aligned[i]) or np.isnan(chop[i]) or
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i])):
            continue
        
        # Long entry conditions
        long_breakout = close[i] > donchian_high[i]
        long_fade = close[i] < bb_lower[i]  # Oversold bounce
        
        # Short entry conditions
        short_breakout = close[i] < donchian_low[i]
        short_fade = close[i] > bb_upper[i]  # Overbought reversal
        
        # Regime filters
        is_trending = adx_weekly_aligned[i] > 25  # Strong trend
        is_chopping = chop[i] > 61.8  # High chopping = ranging market
        
        # Enter long: breakout in trending OR fade in ranging
        if (long_breakout and is_trending) or (long_fade and is_chopping):
            if position <= 0:  # Only enter if not already long
                position = 1
                signals[i] = base_size
        
        # Enter short: breakout in trending OR fade in ranging
        elif (short_breakout and is_trending) or (short_fade and is_chopping):
            if position >= 0:  # Only enter if not already short
                position = -1
                signals[i] = -base_size
        
        # Exit conditions
        elif position == 1:
            # Exit long: opposite breakout or fading signal
            if short_breakout or close[i] > bb_upper[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit short: opposite breakout or fading signal
            if long_breakout or close[i] < bb_lower[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_Choppiness_Donchian_BB_WeeklyTrend"
timeframe = "1d"
leverage = 1.0
#!/usr/bin/env python3
# 6h_weekly_pivot_donchian_breakout_v1
# Hypothesis: 6h strategy using weekly Camarilla pivot levels for structure and 6h Donchian breakout for entry.
# In bull markets: buy breakouts above weekly R3/R4 with volume confirmation.
# In bear markets: sell breakdowns below weekly S3/S4 with volume confirmation.
# Uses weekly HTF pivot to avoid whipsaws and align with major structure.
# Discrete position sizing (0.25) to limit fee drag. Target: 12-37 trades/year.
# Works in both bull and bear by following breakouts aligned with weekly pivot extremes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels: upper=rolling max(high), lower=rolling min(low)"""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels for the period"""
    pivot = (high + low + close) / 3.0
    range_val = high - low
    r3 = pivot + (range_val * 1.1 / 4.0)
    r4 = pivot + (range_val * 1.1 / 2.0)
    s3 = pivot - (range_val * 1.1 / 4.0)
    s4 = pivot - (range_val * 1.1 / 2.0)
    return r3, r4, s3, s4

name = "6h_weekly_pivot_donchian_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 6h Donchian channels (20-period)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    
    # 1d HTF data for weekly Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least a week of daily data
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivots from daily data
    # Group daily data into weeks (starting Monday)
    df_1d_copy = df_1d.copy()
    df_1d_copy['week_start'] = pd.to_datetime(df_1d_copy.index).to_period('W').start_time
    weekly = df_1d_copy.groupby('week_start').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    if len(weekly) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each week
    weekly['r3'], weekly['r4'], weekly['s3'], weekly['s4'] = zip(*weekly.apply(
        lambda row: calculate_camarilla_pivot(row['high'], row['low'], row['close']), axis=1))
    
    # Align weekly data to 1d then to 6h
    # First create daily arrays with weekly values
    daily_r3 = np.full(len(df_1d), np.nan)
    daily_r4 = np.full(len(df_1d), np.nan)
    daily_s3 = np.full(len(df_1d), np.nan)
    daily_s4 = np.full(len(df_1d), np.nan)
    
    # Map weekly values to daily data
    week_start_daily = pd.to_datetime(df_1d.index).to_period('W').start_time
    for idx, week_start in enumerate(weekly['week_start']):
        mask = (week_start_daily == week_start)
        if mask.any():
            daily_r3[mask] = weekly.iloc[idx]['r3']
            daily_r4[mask] = weekly.iloc[idx]['r4']
            daily_s3[mask] = weekly.iloc[idx]['s3']
            daily_s4[mask] = weekly.iloc[idx]['s4']
    
    # Forward fill weekly levels to daily (hold until next week)
    daily_r3 = pd.Series(daily_r3).ffill().values
    daily_r4 = pd.Series(daily_r4).ffill().values
    daily_s3 = pd.Series(daily_s3).ffill().values
    daily_s4 = pd.Series(daily_s4).ffill().values
    
    # Align daily to 6h
    r3_6h = align_htf_to_ltf(prices, df_1d, daily_r3)
    r4_6h = align_htf_to_ltf(prices, df_1d, daily_r4)
    s3_6h = align_htf_to_ltf(prices, df_1d, daily_s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, daily_s4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or np.isnan(r3_6h[i]) or np.isnan(r4_6h[i]) or
            np.isnan(s3_6h[i]) or np.isnan(s4_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below weekly S3 (mean reversion from extreme)
            if close[i] < s3_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above weekly R3 (mean reversion from extreme)
            if close[i] > r3_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter only with volume confirmation
            if volume_confirmed:
                # Long: Donchian breakout above weekly R4 (strong bullish breakout)
                if close[i] > donchian_upper[i] and close[i] > r4_6h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: Donchian breakdown below weekly S4 (strong bearish breakdown)
                elif close[i] < donchian_lower[i] and close[i] < s4_6h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly Donchian(20) breakout with volume confirmation and weekly trend filter
# Uses weekly Donchian channel breakouts for trend capture, daily volume to confirm breakout strength,
# and weekly EMA50 for trend filter. Works in both bull and bear by only taking breakouts in the direction of weekly trend.
# Target: 40-100 total trades over 4 years (10-25/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data (trend filter and Donchian calculation)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Load daily data for volume confirmation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    volume_daily = df_daily['volume'].values
    
    # Calculate weekly Donchian channels (20-period)
    donch_high_weekly = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    donch_low_weekly = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly EMA50 for trend filter
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily volume average (20-period)
    vol_avg_daily = pd.Series(volume_daily).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to daily timeframe
    donch_high_weekly_aligned = align_htf_to_ltf(prices, df_weekly, donch_high_weekly)
    donch_low_weekly_aligned = align_htf_to_ltf(prices, df_weekly, donch_low_weekly)
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    vol_avg_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_daily)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_weekly_aligned[i]) or np.isnan(donch_low_weekly_aligned[i]) or
            np.isnan(ema50_weekly_aligned[i]) or np.isnan(vol_avg_daily_aligned[i])):
            continue
        
        # Long entry: price breaks above weekly Donchian high + volume spike + price above weekly EMA50
        if (close[i] > donch_high_weekly_aligned[i] and
            volume[i] > 1.5 * vol_avg_daily_aligned[i] and
            close[i] > ema50_weekly_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below weekly Donchian low + volume spike + price below weekly EMA50
        elif (close[i] < donch_low_weekly_aligned[i] and
              volume[i] > 1.5 * vol_avg_daily_aligned[i] and
              close[i] < ema50_weekly_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal
        elif position == 1 and close[i] < donch_low_weekly_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > donch_high_weekly_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyDonchian_Volume_Trend_Filter"
timeframe = "1d"
leverage = 1.0
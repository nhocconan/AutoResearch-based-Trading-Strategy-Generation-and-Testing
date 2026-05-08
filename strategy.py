#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian Breakout with Weekly Trend Filter and Volume Confirmation
# Uses weekly Donchian trend for bias, 12h Donchian breakout for entry, and volume surge (>1.5x average) for confirmation.
# Designed to capture strong trends in both bull and bear markets while avoiding false breakouts in low-volume conditions.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years).

name = "12h_DonchianBreakout_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Weekly upper band (20-period high)
    donchian_high_weekly = np.full(len(high_weekly), np.nan)
    for i in range(20, len(high_weekly)):
        donchian_high_weekly[i] = np.max(high_weekly[i-20:i])
    
    # Weekly lower band (20-period low)
    donchian_low_weekly = np.full(len(low_weekly), np.nan)
    for i in range(20, len(low_weekly)):
        donchian_low_weekly[i] = np.min(low_weekly[i-20:i])
    
    # Get daily data for volume average
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily volume average (20-period)
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    for i in range(20, len(vol_daily)):
        vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
    # Align weekly Donchian bands to 12h timeframe
    donchian_high_weekly_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high_weekly)
    donchian_low_weekly_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low_weekly)
    
    # Align daily volume average to 12h timeframe
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
    # Calculate 12h Donchian breakout levels (20-period)
    donchian_high_12h = np.full(n, np.nan)
    donchian_low_12h = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high_12h[i] = np.max(high[i-20:i])
        donchian_low_12h[i] = np.min(low[i-20:i])
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_weekly_aligned[i]) or np.isnan(donchian_low_weekly_aligned[i]) or
            np.isnan(vol_avg_20_daily_aligned[i]) or np.isnan(donchian_high_12h[i]) or np.isnan(donchian_low_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average of daily volume
        vol_confirmation = volume[i] > 1.5 * vol_avg_20_daily_aligned[i]
        
        if position == 0:
            # Look for entry: breakout above weekly resistance in uptrend or below weekly support in downtrend
            # Uptrend: price above weekly Donchian middle (bullish bias)
            weekly_mid = (donchian_high_weekly_aligned[i] + donchian_low_weekly_aligned[i]) / 2
            uptrend = close[i] > weekly_mid
            
            # Downtrend: price below weekly Donchian middle (bearish bias)
            downtrend = close[i] < weekly_mid
            
            # Long: price breaks above 12h Donchian resistance in uptrend with volume confirmation
            long_condition = (
                close[i] > donchian_high_12h[i] and   # break above 12h resistance
                uptrend and                           # weekly uptrend bias
                vol_confirmation                      # volume confirmation
            )
            
            # Short: price breaks below 12h Donchian support in downtrend with volume confirmation
            short_condition = (
                close[i] < donchian_low_12h[i] and    # break below 12h support
                downtrend and                         # weekly downtrend bias
                vol_confirmation                      # volume confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below 12h Donchian support or weekly trend turns down
            if close[i] < donchian_low_12h[i] or close[i] < weekly_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above 12h Donchian resistance or weekly trend turns up
            if close[i] > donchian_high_12h[i] or close[i] > weekly_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
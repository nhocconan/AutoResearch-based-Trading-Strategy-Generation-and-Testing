#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Donchian(20) breakout with daily trend filter and volume confirmation
# Long when price breaks above weekly Donchian upper(20) AND 1d EMA34 > EMA34 previous (uptrend) AND volume > 1.8 * avg_volume(20) on 6h
# Short when price breaks below weekly Donchian lower(20) AND 1d EMA34 < EMA34 previous (downtrend) AND volume > 1.8 * avg_volume(20) on 6h
# Exit when price crosses weekly Donchian midpoint (mean reversion)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Weekly Donchian provides structure from higher timeframe, reducing noise
# Daily EMA34 trend filter ensures we trade with the dominant daily trend
# Volume spike confirmation validates breakout strength while limiting overtrading
# Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)

name = "6h_WeeklyDonchian20_1dEMA34_Trend_VolumeConfirm"
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
    
    # Get weekly data ONCE before loop for Donchian calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 completed weekly bars for Donchian(20)
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels: 20-period high/low
    highest_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_upper_20 = highest_high_20
    donchian_lower_20 = lowest_low_20
    donchian_mid_20 = (donchian_upper_20 + donchian_lower_20) / 2.0
    
    # Align weekly Donchian to 6h timeframe (wait for completed weekly bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid_20)
    
    # Get daily data ONCE before loop for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 completed daily bars for EMA34
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume confirmation: volume > 1.8 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian upper(20), 1d EMA34 > EMA34 previous (uptrend), volume spike, in session
            if (close[i] > donchian_upper_aligned[i] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian lower(20), 1d EMA34 < EMA34 previous (downtrend), volume spike, in session
            elif (close[i] < donchian_lower_aligned[i] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly Donchian midpoint (mean reversion)
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above weekly Donchian midpoint (mean reversion)
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
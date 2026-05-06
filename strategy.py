#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly ATR-based Donchian breakout with volume confirmation and close-based exit
# Long when price breaks above weekly Donchian high (20-period) with volume > 1.5x 50-day average
# Short when price breaks below weekly Donchian low (20-period) with volume > 1.5x 50-day average
# Exit on close crossing below/above weekly Donchian midpoint
# Weekly Donchian provides structural levels, volume confirms institutional interest
# Target: 15-25 trades per year (60-100 over 4 years) with 0.25 position sizing

name = "1d_weeklyDonchian20_Volume_ExitMidpoint"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data (HTF)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # 20-period rolling max/min on weekly data
    weekly_donchian_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    weekly_donchian_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    weekly_donchian_mid = (weekly_donchian_high + weekly_donchian_low) / 2
    
    # Align weekly Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, weekly_donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, weekly_donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_weekly, weekly_donchian_mid)
    
    # Volume confirmation: >1.5x 50-day average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (1.5 * vol_ma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after volume MA warmup
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above weekly Donchian high with volume confirmation
            if close[i] > donchian_high_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below weekly Donchian low with volume confirmation
            elif close[i] < donchian_low_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: close crosses below weekly Donchian midpoint
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: close crosses above weekly Donchian midpoint
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
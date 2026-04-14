#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Donchian(20) breakout with daily volume confirmation
# Long when daily price breaks above weekly Donchian upper band with daily volume >1.3x 20-day average
# Short when daily price breaks below weekly Donchian lower band with daily volume >1.3x 20-day average
# Exit when price crosses the weekly Donchian midline
# Weekly trend filter to avoid counter-trend trades in ranging markets
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channel (20-period lookback)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_upper = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate daily volume average (20-period)
    vol_ma_20d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly indicators to daily timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1w, donchian_middle)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 60  # for 20-period calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma_20d[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long setup: break above weekly Donchian upper with volume confirmation
            if (price > donchian_upper_aligned[i] and 
                vol_current > 1.3 * vol_ma_20d[i]):
                position = 1
                signals[i] = position_size
            # Short setup: break below weekly Donchian lower with volume confirmation
            elif (price < donchian_lower_aligned[i] and 
                  vol_current > 1.3 * vol_ma_20d[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below weekly Donchian middle
            if price < donchian_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above weekly Donchian middle
            if price > donchian_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_WeeklyDonchian_Volume"
timeframe = "1d"
leverage = 1.0
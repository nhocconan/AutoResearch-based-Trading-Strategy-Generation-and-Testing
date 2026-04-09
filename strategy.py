#!/usr/bin/env python3
# 1d_weekly_donchian_breakout_v1
# Hypothesis: Weekly Donchian breakout on daily chart with volume confirmation works in both bull and bear markets.
# Uses weekly Donchian channels (20-period) to identify trend direction, enters on daily breakouts in trend direction.
# Includes volume confirmation (>1.5x 20-day average) and exits on opposite breakout or volume drop.
# Position size 0.25. Target: 15-25 trades/year (60-100 over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    week_high = df_weekly['high'].values
    week_low = df_weekly['low'].values
    week_donchian_high = pd.Series(week_high).rolling(window=20, min_periods=20).max().values
    week_donchian_low = pd.Series(week_low).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe (properly delayed for weekly close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, week_donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, week_donchian_low)
    
    # Volume confirmation: 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below weekly Donchian low OR volume drops below average
            if close[i] <= donchian_low_aligned[i] or volume[i] < vol_ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above weekly Donchian high OR volume drops below average
            if close[i] >= donchian_high_aligned[i] or volume[i] < vol_ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above weekly Donchian high with volume confirmation
            if close[i] > donchian_high_aligned[i] and volume[i] > vol_threshold[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below weekly Donchian low with volume confirmation
            elif close[i] < donchian_low_aligned[i] and volume[i] > vol_threshold[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
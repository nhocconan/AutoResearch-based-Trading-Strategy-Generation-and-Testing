#!/usr/bin/env python3
# 12h_donchian_1w_trend_volume_v1
# Hypothesis: 12h strategy using 1w Donchian breakouts for trend direction, volume confirmation for entry timing.
# Long: price > 1w Donchian upper (20) + volume > 2.0x 20-period average
# Short: price < 1w Donchian lower (20) + volume > 2.0x 20-period average
# Exit: price crosses 1w Donchian midpoint (mean reversion) OR opposite Donchian breakout
# Uses discrete sizing (±0.25) to minimize fee churn. Target: 50-150 total trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1w Donchian channels (20-period)
    period20_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_upper = period20_high
    donchian_lower = period20_low
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Align Donchian channels to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below midpoint OR breaks below lower Donchian (strong reversal)
            if close[i] < donchian_mid_aligned[i] or close[i] < donchian_lower_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above midpoint OR breaks above upper Donchian (strong reversal)
            if close[i] > donchian_mid_aligned[i] or close[i] > donchian_upper_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price breaks above upper Donchian
                if close[i] > donchian_upper_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below lower Donchian
                elif close[i] < donchian_lower_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
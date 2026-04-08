#!/usr/bin/env python3
# 1d_weekly_price_action_v1
# Hypothesis: Uses weekly Donchian channels for trend direction and daily price action for entry.
# Long when: daily close above weekly Donchian upper band (20) and volume > 1.5x 20-day average.
# Short when: daily close below weekly Donchian lower band (20) and volume > 1.5x 20-day average.
# Exit when price crosses back inside the weekly Donchian channel.
# Uses weekly trend filter to avoid counter-trend trades, reducing whipsaw in sideways markets.
# Target: 15-25 trades/year per symbol.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_price_action_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: 1.5x 20-day average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get weekly data for Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Weekly Donchian channel (20-period)
    donchian_period = 20
    upper_weekly = np.full(len(high_weekly), np.nan)
    lower_weekly = np.full(len(low_weekly), np.nan)
    
    for i in range(donchian_period-1, len(high_weekly)):
        upper_weekly[i] = np.max(high_weekly[i-donchian_period+1:i+1])
        lower_weekly[i] = np.min(low_weekly[i-donchian_period+1:i+1])
    
    # Align weekly Donchian levels to daily timeframe
    upper_weekly_aligned = align_htf_to_ltf(prices, df_weekly, upper_weekly)
    lower_weekly_aligned = align_htf_to_ltf(prices, df_weekly, lower_weekly)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(vol_ma_period, donchian_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_ma[i]) or np.isnan(upper_weekly_aligned[i]) or np.isnan(lower_weekly_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below weekly Donchian upper band
            if close[i] < upper_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above weekly Donchian lower band
            if close[i] > lower_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above weekly Donchian upper band with volume surge
            if close[i] > upper_weekly_aligned[i] and vol_surge[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Price below weekly Donchian lower band with volume surge
            elif close[i] < lower_weekly_aligned[i] and vol_surge[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
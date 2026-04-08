#!/usr/bin/env python3
# 4h_donchian_breakout_1d_trend_volume_v2
# Hypothesis: 4-hour Donchian breakout with 1-day trend filter and volume confirmation.
# Long when price breaks above 20-period Donchian high with 1-day uptrend and volume spike.
# Short when price breaks below 20-period Donchian low with 1-day downtrend and volume spike.
# Exit when price returns to the 10-period Donchian midpoint or opposite signal.
# Designed to capture trend continuation in both bull and bear markets with low trade frequency.
# Target: 20-40 trades/year to minimize fee drift while capturing strong momentum moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 20-period Donchian channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Get 1-day data for trend filter (calculate once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1-day trend: close > open = uptrend, close < open = downtrend
    open_1d = df_1d['open'].values
    close_1d = df_1d['close'].values
    daily_uptrend = close_1d > open_1d
    daily_downtrend = close_1d < open_1d
    
    # Align 1-day trend to 4-hour chart
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    # Volume confirmation: 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or \
           np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i]) or \
           np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to midpoint or opposite signal
            if close[i] <= donchian_mid[i] or \
               (close[i] >= donchian_high[i] and volume[i] > 1.5 * avg_volume[i] and daily_downtrend_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to midpoint or opposite signal
            if close[i] >= donchian_mid[i] or \
               (close[i] <= donchian_low[i] and volume[i] > 1.5 * avg_volume[i] and daily_uptrend_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: price breaks above Donchian high with volume and daily uptrend
            if close[i] > donchian_high[i] and volume_ok and daily_uptrend_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume and daily downtrend
            elif close[i] < donchian_low[i] and volume_ok and daily_downtrend_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout direction with 1h volume confirmation and session filter (08-20 UTC).
# Long when 1h price breaks above 4h Donchian upper (20) AND volume > 1.5 * avg_volume(20) AND hour in [08,20] UTC.
# Short when 1h price breaks below 4h Donchian lower (20) AND volume > 1.5 * avg_volume(20) AND hour in [08,20] UTC.
# Exit when price touches 4h Donchian middle.
# Uses discrete sizing 0.20 to minimize fee churn and control drawdown.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
# 4h Donchian provides structural breakout levels with lower frequency.
# Volume confirmation filters weak breakouts.
# Session filter reduces noise during low-liquidity hours.
# Works in bull (trend continuation breakouts) and bear (trend continuation breakdowns).

name = "1h_4hDonchian20_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need sufficient data for Donchian(20)
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_series_4h = pd.Series(high_4h)
    low_series_4h = pd.Series(low_4h)
    donchian_upper_4h = high_series_4h.rolling(window=20, min_periods=20).max().values
    donchian_lower_4h = low_series_4h.rolling(window=20, min_periods=20).min().values
    donchian_middle_4h = (donchian_upper_4h + donchian_lower_4h) / 2.0
    
    # Align 4h Donchian levels to 1h timeframe (wait for completed 4h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle_4h)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Pre-compute session filter (08-20 UTC) - open_time is already datetime64[ms]
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper with volume confirmation and in session
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                volume_confirm[i] and in_session[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian lower with volume confirmation and in session
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  volume_confirm[i] and in_session[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price touches 4h Donchian middle (profit take or reversal)
            if close[i] <= donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price touches 4h Donchian middle (profit take or reversal)
            if close[i] >= donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
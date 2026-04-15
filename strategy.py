#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Donchian Breakout with Volume Confirmation
# Uses weekly price channel breakouts (20-period Donchian) confirmed by volume spike.
# Works in bull markets (breakouts above upper band) and bear markets (breakouts below lower band).
# Target: 30-100 total trades over 4 years (7-25/year). Timeframe: 1d, HTF: 1w.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Donchian channels (weekly calculation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period Donchian channels on 1d (weekly bars)
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 1d timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i])):
            continue
        
        # Long entry: price breaks above weekly Donchian high + volume confirmation
        if (close[i] > donch_high_aligned[i] and
            volume[i] > 2.0 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below weekly Donchian low + volume confirmation
        elif (close[i] < donch_low_aligned[i] and
              volume[i] > 2.0 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout
        elif position == 1 and close[i] < donch_low_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > donch_high_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyDonchian20_Volume"
timeframe = "1d"
leverage = 1.0
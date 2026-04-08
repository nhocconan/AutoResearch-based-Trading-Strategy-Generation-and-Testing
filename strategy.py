#!/usr/bin/env python3
# 6h_weekly_donchian_breakout_volume_v2
# Hypothesis: Weekly Donchian channel breakout with volume confirmation on 6h timeframe.
# Long: Price breaks above weekly Donchian high (20-period) AND volume > 1.5x 20-period average volume
# Short: Price breaks below weekly Donchian low (20-period) AND volume > 1.5x 20-period average volume
# Exit: Opposite breakout or price returns to weekly Donchian midpoint
# Uses 6h primary timeframe with 1w HTF for Donchian channels and volume average.
# Target: 50-150 total trades over 4 years to minimize fee drag and avoid overtrading.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_donchian_breakout_volume_v2"
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
    
    # Get 1w data for Donchian channels and volume average
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly Donchian channels (20-period)
    def rolling_max(arr, window):
        return np.convolve(arr, np.ones(window), 'valid') / window  # placeholder, will replace with proper rolling
    
    # Proper rolling max/min with min_periods
    high_1w_series = pd.Series(high_1w)
    low_1w_series = pd.Series(low_1w)
    volume_1w_series = pd.Series(volume_1w)
    
    donchian_high = high_1w_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_1w_series.rolling(window=20, min_periods=20).min().values
    avg_volume = volume_1w_series.rolling(window=20, min_periods=20).mean().values
    
    # Align weekly indicators to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    avg_volume_aligned = align_htf_to_ltf(prices, df_1w, avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(avg_volume_aligned[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price returns to weekly Donchian midpoint or opposite breakout
            midpoint = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if close[i] <= midpoint or (close[i] < donchian_low_aligned[i] and volume_confirmed):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to weekly Donchian midpoint or opposite breakout
            midpoint = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if close[i] >= midpoint or (close[i] > donchian_high_aligned[i] and volume_confirmed):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: price breaks above weekly Donchian high with volume confirmation
            if close[i] > donchian_high_aligned[i] and volume_confirmed:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below weekly Donchian low with volume confirmation
            elif close[i] < donchian_low_aligned[i] and volume_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
# 6h_weekly_donchian_breakout_volume_v2
# Hypothesis: Weekly Donchian channel breakout with volume confirmation on 6h timeframe.
# Long: Price breaks above weekly Donchian high (20-period) AND volume > 1.5x 20-period average volume
# Short: Price breaks below weekly Donchian low (20-period) AND volume > 1.5x 20-period average volume
# Exit: Opposite breakout or price returns to weekly Donchian midpoint
# Uses 6h primary timeframe with 1w HTF for Donchian channels and volume average.
# Target: 50-150 total trades over 4 years to minimize fee drag and avoid overtrading.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_donchian_breakout_volume_v2"
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
    
    # Get 1w data for Donchian channels and volume average
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly Donchian channels (20-period)
    high_1w_series = pd.Series(high_1w)
    low_1w_series = pd.Series(low_1w)
    volume_1w_series = pd.Series(volume_1w)
    
    donchian_high = high_1w_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_1w_series.rolling(window=20, min_periods=20).min().values
    avg_volume = volume_1w_series.rolling(window=20, min_periods=20).mean().values
    
    # Align weekly indicators to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    avg_volume_aligned = align_htf_to_ltf(prices, df_1w, avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(avg_volume_aligned[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price returns to weekly Donchian midpoint or opposite breakout
            midpoint = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if close[i] <= midpoint or (close[i] < donchian_low_aligned[i] and volume_confirmed):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to weekly Donchian midpoint or opposite breakout
            midpoint = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if close[i] >= midpoint or (close[i] > donchian_high_aligned[i] and volume_confirmed):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: price breaks above weekly Donchian high with volume confirmation
            if close[i] > donchian_high_aligned[i] and volume_confirmed:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below weekly Donchian low with volume confirmation
            elif close[i] < donchian_low_aligned[i] and volume_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals
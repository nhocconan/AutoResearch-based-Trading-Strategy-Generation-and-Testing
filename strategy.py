#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d trend filter and volume confirmation
# Long when price breaks above Donchian(20) upper band AND price > 1d EMA(50) AND volume > 1.5x 20-period average
# Short when price breaks below Donchian(20) lower band AND price < 1d EMA(50) AND volume > 1.5x 20-period average
# Exit when price crosses back through Donchian center (mean of upper/lower bands)
# Uses Donchian channels for breakout signals, EMA for trend filter, volume for confirmation
# Target: 50-150 total trades over 4 years (12-37/year) for optimal 12h performance

name = "12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # 1d EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    one_day_close = df_1d['close'].values
    one_day_close_series = pd.Series(one_day_close)
    one_day_ema = one_day_close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    one_day_ema_aligned = align_htf_to_ltf(prices, df_1d, one_day_ema)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or np.isnan(one_day_ema_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price crosses Donchian middle
        if position == 1:  # long position
            if close[i] <= donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: price breaks above Donchian upper AND price > 1d EMA(50) AND volume confirmation
            if (close[i] > donchian_upper[i] and close[i] > one_day_ema_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND price < 1d EMA(50) AND volume confirmation
            elif (close[i] < donchian_lower[i] and close[i] < one_day_ema_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals
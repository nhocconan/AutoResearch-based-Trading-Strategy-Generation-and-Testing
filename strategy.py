#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with 1d volume confirmation
# - Use 4h Donchian channels (20-period) for trend direction and breakout signals
# - Enter long when price breaks above 4h upper band with 1d volume confirmation
# - Enter short when price breaks below 4h lower band with 1d volume confirmation
# - Use 1h only for entry timing precision, 4h for direction, 1d for volume filter
# - Target: 60-150 total trades over 4 years (15-37/year) with 0.20 position sizing
# - Session filter (08-20 UTC) to reduce noise trades

name = "1h_Donchian20_4hTrend_1dVolume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period high/low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Rolling max/min for Donchian channels
    upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume moving average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1h volume > 1.5 * 1d volume MA
        volume_confirm = volume[i] > (1.5 * vol_ma_1d_aligned[i])
        
        if position == 0:
            # Long entry: price breaks above 4h upper band with volume confirmation
            if close[i] > upper_4h_aligned[i] and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below 4h lower band with volume confirmation
            elif close[i] < lower_4h_aligned[i] and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to 4h midpoint or breaks below lower band
            midpoint = (upper_4h_aligned[i] + lower_4h_aligned[i]) / 2
            if close[i] < lower_4h_aligned[i]:  # Stop loss
                signals[i] = 0.0
                position = 0
            elif close[i] < midpoint:  # Take profit at midpoint
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to 4h midpoint or breaks above upper band
            midpoint = (upper_4h_aligned[i] + lower_4h_aligned[i]) / 2
            if close[i] > upper_4h_aligned[i]:  # Stop loss
                signals[i] = 0.0
                position = 0
            elif close[i] > midpoint:  # Take profit at midpoint
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
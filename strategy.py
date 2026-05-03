#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and 1d volume spike.
# Long when price breaks above Donchian upper band AND price > 1w EMA50 AND 1d volume > 1.5x 20-period MA.
# Short when price breaks below Donchian lower band AND price < 1w EMA50 AND 1d volume > 1.5x 20-period MA.
# Exit when price touches Donchian middle band (mean of upper/lower) OR volume drops below average.
# Uses 12h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Donchian channels provide clear structure, 1w EMA50 filters for higher-timeframe trend, 1d volume confirms participation.
# Designed to work in both bull (breakouts above EMA50) and bear (breakdowns below EMA50) markets.

name = "12h_Donchian20_1wEMA50_1dVolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume 20-period MA
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 12h Donchian channels (20-period)
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_rolling_max
    donchian_lower = low_rolling_min
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Volume spike condition: current 12h volume > 1.5x 1d volume 20-period MA
        volume_spike = volume[i] > (volume_ma_20_1d_aligned[i] * 1.5)
        
        if position == 0:
            # Long: price breaks above Donchian upper AND price > 1w EMA50 AND volume spike AND session
            if close[i] > donchian_upper[i] and close[i] > ema_50_1w_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND price < 1w EMA50 AND volume spike AND session
            elif close[i] < donchian_lower[i] and close[i] < ema_50_1w_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches Donchian middle OR volume drops below average
            if close[i] <= donchian_middle[i] or volume[i] < volume_ma_20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches Donchian middle OR volume drops below average
            if close[i] >= donchian_middle[i] or volume[i] < volume_ma_20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
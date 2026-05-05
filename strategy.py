#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above 12h Donchian upper(20) AND price > 1d EMA34 (uptrend) AND volume > 1.3x 20-period average
# Short when price breaks below 12h Donchian lower(20) AND price < 1d EMA34 (downtrend) AND volume > 1.3x 20-period average
# Exit when price crosses 12h Donchian midpoint (middle of channel)
# Uses Donchian channels for clear breakout levels, 1d EMA34 for trend filter (more responsive than EMA50)
# Volume confirmation ensures institutional participation (moderate threshold to minimize trades)
# Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# Timeframe: 12h (primary timeframe as required)
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "12h_Donchian20_1dEMA34_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian(20) channels
    # Upper = max(high_12h, 20), Lower = min(low_12h, 20)
    # Using rolling window with min_periods
    high_series = pd.Series(high_12h)
    low_series = pd.Series(low_12h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Get 1d data ONCE before loop for EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_12h, donchian_middle)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation on 12h (threshold: 1.3x for tighter filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.3 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper AND price > EMA34 (uptrend) AND volume spike
            if (close[i] > donchian_upper_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND price < EMA34 (downtrend) AND volume spike
            elif (close[i] < donchian_lower_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian middle
            if close[i] < donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian middle
            if close[i] > donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
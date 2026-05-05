#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above 1d Donchian upper(20) AND price > 1w EMA34 (uptrend) AND volume > 2.0x 20-period average
# Short when price breaks below 1d Donchian lower(20) AND price < 1w EMA34 (downtrend) AND volume > 2.0x 20-period average
# Exit when price crosses 1d Donchian midpoint OR EMA34 filter reverses
# Uses Donchian channels for clear trend structure, 1w EMA34 for regime filter (avoid whipsaws in chop)
# Volume spike confirms institutional participation
# Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# Timeframe: 1d (primary timeframe as required)
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag

name = "1d_Donchian20_1wEMA34_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    # Upper = max(high, 20), Lower = min(low, 20), Mid = (Upper + Lower) / 2
    high_rolling_max = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    upper_20 = high_rolling_max
    lower_20 = low_rolling_min
    midpoint_20 = (upper_20 + lower_20) / 2.0
    
    # Get 1w data ONCE before loop for EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(34)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 1d timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    midpoint_aligned = align_htf_to_ltf(prices, df_1d, midpoint_20)
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation on 1d (threshold: 2.0x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(midpoint_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper(20) AND price > EMA34 (uptrend) AND volume spike
            if (close[i] > upper_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower(20) AND price < EMA34 (downtrend) AND volume spike
            elif (close[i] < lower_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below midpoint OR price < EMA34 (trend weakening)
            if close[i] < midpoint_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above midpoint OR price > EMA34 (trend weakening)
            if close[i] > midpoint_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
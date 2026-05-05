#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA34 trend filter + volume confirmation
# Long when price breaks above 1d Donchian upper AND 1w EMA34 > price (uptrend) AND volume spike
# Short when price breaks below 1d Donchian lower AND 1w EMA34 < price (downtrend) AND volume spike
# Exit when price crosses the 1d Donchian middle (mean) OR trend reverses (EMA34 crosses price)
# Uses Donchian channels for structure, 1w EMA34 for trend filter (avoid whipsaws in ranging markets)
# Volume spike confirms institutional participation at breakouts
# Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# Timeframe: 1d (primary timeframe as required)
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag

name = "1d_Donchian20_1wEMA34_Trend_VolumeSpike"
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
    
    # Get 1d data ONCE before loop for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian(20) channels
    # Upper = max(high, 20)
    # Lower = min(low, 20)
    # Middle = (upper + lower) / 2
    high_ma_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma_20
    donchian_lower = low_ma_20
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(34)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 1d timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
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
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper AND EMA34 > price (uptrend) AND volume spike
            if (close[i] > donchian_upper_aligned[i] and 
                ema_34_1w_aligned[i] > close[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND EMA34 < price (downtrend) AND volume spike
            elif (close[i] < donchian_lower_aligned[i] and 
                  ema_34_1w_aligned[i] < close[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian middle OR trend reverses (EMA34 < price)
            if close[i] < donchian_middle_aligned[i] or ema_34_1w_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian middle OR trend reverses (EMA34 > price)
            if close[i] > donchian_middle_aligned[i] or ema_34_1w_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
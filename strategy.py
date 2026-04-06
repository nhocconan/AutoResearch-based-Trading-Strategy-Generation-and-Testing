#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume confirmation + 1d EMA(20) trend filter
# Enter long when price breaks above 12h Donchian upper(20), volume > 1.5x 20-day average, and 1d EMA(20) rising
# Enter short when price breaks below 12h Donchian lower(20), volume > 1.5x 20-day average, and 1d EMA(20) falling
# Exit when price returns to 12h Donchian median or opposite breakout occurs
# Target: 50-150 trades over 4 years by using 12h timeframe with multiple confluence filters

name = "12h_donchian_1d_ema_vol_v1"
timeframe = "12h"
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
    
    # 12h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 1d EMA(20) trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False).mean().values
    ema_20_rising = ema_20 > np.roll(ema_20, 1)
    ema_20_falling = ema_20 < np.roll(ema_20, 1)
    ema_20_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_20_rising)
    ema_20_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_20_falling)
    
    # 1d volume confirmation: volume > 1.5x 20-day average
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_threshold = 1.5 * vol_ma_20
    vol_threshold_aligned = align_htf_to_ltf(prices, df_1d, vol_threshold)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_20_rising_aligned[i]) or np.isnan(ema_20_falling_aligned[i]) or
            np.isnan(vol_threshold_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price below Donchian median OR opposite breakout
            if close[i] < donchian_mid[i] or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above Donchian median OR opposite breakout
            if close[i] > donchian_mid[i] or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + EMA trend
            vol_ok = volume[i] > vol_threshold_aligned[i]
            
            if vol_ok and ema_20_rising_aligned[i]:
                # Bullish: price breaks above Donchian upper with rising EMA
                if close[i] > donchian_high[i] and (i == 0 or close[i-1] <= donchian_high[i-1]):
                    signals[i] = 0.25
                    position = 1
            elif vol_ok and ema_20_falling_aligned[i]:
                # Bearish: price breaks below Donchian lower with falling EMA
                if close[i] < donchian_low[i] and (i == 0 or close[i-1] >= donchian_low[i-1]):
                    signals[i] = -0.25
                    position = -1
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d Williams %R extreme filter and volume confirmation.
# Long: Close breaks above Donchian upper AND 1d Williams %R < -80 (oversold) AND volume > 1.5x 20-period MA
# Short: Close breaks below Donchian lower AND 1d Williams %R > -20 (overbought) AND volume > 1.5x 20-period MA
# Exit: Opposite Donchian breakout or Williams %R returns to neutral range (-80 to -20) or volume drops.
# Discrete sizing 0.25. Target: 75-200 total trades over 4 years (19-50/year).
# Donchian provides clear structure; 1d Williams %R filters for mean reversion extremes in higher timeframe;
# volume confirmation reduces false breakouts. Works in bull via long signals from oversold bounces and bear via short signals from overbought reversals.

name = "4h_Donchian20_1dWilliamsR_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14-period)
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(df_1d_high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d_low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = -100 * (highest_high - df_1d_close) / (highest_high - lowest_low + 1e-10)
    
    # Align 1d Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Donchian channels (20-period) on 4h
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: current 4h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        williams_r_val = williams_r_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine Williams %R regime
        is_oversold = williams_r_val < -80
        is_overbought = williams_r_val > -20
        is_neutral = (williams_r_val >= -80) & (williams_r_val <= -20)
        
        # Entry logic
        if position == 0:
            # Long: Close breaks above Donchian upper AND oversold AND volume spike
            if close_val > donchian_upper[i] and is_oversold and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian lower AND overbought AND volume spike
            elif close_val < donchian_lower[i] and is_overbought and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close breaks below Donchian lower OR Williams %R returns to neutral OR volume drops
            if close_val < donchian_lower[i] or is_neutral or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close breaks above Donchian upper OR Williams %R returns to neutral OR volume drops
            if close_val > donchian_upper[i] or is_neutral or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
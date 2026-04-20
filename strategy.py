#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d chart with 1w Williams %R filter and 1d Donchian breakout.
# Long when price breaks above Donchian(20) high AND Williams %R > -20 (not oversold).
# Short when price breaks below Donchian(20) low AND Williams %R < -80 (not overbought).
# Uses weekly Williams %R to filter out counter-trend extremes and avoid false breakouts.
# Target: 15-30 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian channels: 20-period high/low
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Previous day's Donchian levels (to avoid look-ahead)
    prev_donch_high = np.roll(donch_high, 1)
    prev_donch_low = np.roll(donch_low, 1)
    prev_donch_high[0] = high_1d[0]
    prev_donch_low[0] = low_1d[0]
    
    # Align Donchian levels to 1d timeframe (they're already 1d, but align for consistency)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, prev_donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, prev_donch_low)
    
    # Load 1w data for Williams %R
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1w) / (highest_high - lowest_low) * -100
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 1d timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    
    # 1d data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.3x 20-period average (milder filter)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20) > 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(williams_r_aligned[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        williams_r_val = williams_r_aligned[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND Williams %R > -20 (not oversold) AND volume
            if price > donch_high_val and williams_r_val > -20 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND Williams %R < -80 (not overbought) AND volume
            elif price < donch_low_val and williams_r_val < -80 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low OR Williams %R < -80 (overbought)
            if price < donch_low_val or williams_r_val < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high OR Williams %R > -20 (oversold)
            if price > donch_high_val or williams_r_val > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_WilliamsR_Donchian_Breakout_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0
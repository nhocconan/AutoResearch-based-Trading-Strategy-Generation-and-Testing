#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian(20) breakout with 1-week trend filter and volume confirmation
# In bull market (close > weekly EMA50): buy breakout above 20-day high
# In bear market (close < weekly EMA50): sell breakdown below 20-day low
# Volume confirmation: require volume > 1.5x 20-day average to filter false breakouts
# Designed to work in both bull and bear markets by adapting to trend direction
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 50-period EMA on weekly timeframe for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Load daily data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels on daily timeframe
    highest_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume filter: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume_1d > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if NaN in indicators
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market trend from weekly EMA50
        is_bull = close_1d[i] > ema50_1w_aligned[i]
        is_bear = close_1d[i] < ema50_1w_aligned[i]
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        price = close_1d[i]
        
        if position == 0:
            # Enter long: breakout above 20-day high in bull market
            long_signal = False
            if has_volume and is_bull:
                if price > highest_high_20[i]:
                    long_signal = True
            
            # Enter short: breakdown below 20-day low in bear market
            short_signal = False
            if has_volume and is_bear:
                if price < lowest_low_20[i]:
                    short_signal = True
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 20-day low or trend changes to bear
            exit_signal = False
            if price < lowest_low_20[i] or is_bear:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 20-day high or trend changes to bull
            exit_signal = False
            if price > highest_high_20[i] or is_bull:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_WeeklyTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0
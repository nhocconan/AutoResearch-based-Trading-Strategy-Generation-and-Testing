#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian channel breakout with weekly trend filter and volume confirmation
# Long when price breaks above 20-day high AND weekly EMA20 is rising AND volume > 1.5x average
# Short when price breaks below 20-day low AND weekly EMA20 is falling AND volume > 1.5x average
# Uses discrete position sizing (0.25) to limit risk and reduce trade frequency
# Designed to capture strong trends while avoiding choppy markets
# Target: 20-60 total trades over 4 years (5-15/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels on daily timeframe
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume average (20-period)
    volume_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Load weekly data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 20-period EMA on weekly timeframe
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly EMA to daily timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in indicators
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(volume_ma[i]) or np.isnan(ema20_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        volume = volume_1d[i]
        
        if position == 0:
            # Enter long conditions
            long_signal = (
                price > donchian_high[i] and  # Break above 20-day high
                ema20_1w_aligned[i] > ema20_1w_aligned[i-1] and  # Weekly EMA20 rising
                volume > 1.5 * volume_ma[i]  # Volume > 1.5x average
            )
            
            # Enter short conditions
            short_signal = (
                price < donchian_low[i] and  # Break below 20-day low
                ema20_1w_aligned[i] < ema20_1w_aligned[i-1] and  # Weekly EMA20 falling
                volume > 1.5 * volume_ma[i]  # Volume > 1.5x average
            )
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low or weekly EMA turns down
            exit_signal = (
                price < donchian_low[i] or
                ema20_1w_aligned[i] < ema20_1w_aligned[i-1]
            )
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high or weekly EMA turns up
            exit_signal = (
                price > donchian_high[i] or
                ema20_1w_aligned[i] > ema20_1w_aligned[i-1]
            )
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_WeeklyEMA20_VolumeConfirm"
timeframe = "1d"
leverage = 1.0
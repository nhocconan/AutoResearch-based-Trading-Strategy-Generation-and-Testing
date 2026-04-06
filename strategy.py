#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily breakout with weekly trend filter and volume confirmation
# Enter long when: price breaks above 1w Donchian(20) high, volume > 1.5x 20-day avg, price > 1w EMA(20)
# Enter short when: price breaks below 1w Donchian(20) low, volume > 1.5x 20-day avg, price < 1w EMA(20)
# Exit when price re-enters Donchian channel or opposite breakout occurs
# Uses weekly structure to capture multi-week trends with daily precision, targeting 40-80 trades over 4 years

name = "1d_weekly_donchian20_ema20_vol_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly Donchian(20) for breakout levels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 20-period Donchian channels on weekly data
    high_max = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly channels to daily timeframe
    donchian_high = align_htf_to_ltf(prices, df_1w, high_max)
    donchian_low = align_htf_to_ltf(prices, df_1w, low_min)
    
    # Weekly EMA(20) for trend filter
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Volume confirmation: volume > 1.5x 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for indicators to stabilize
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price re-enters Donchian channel OR breaks below weekly EMA(20)
            if close[i] <= donchian_high[i] or close[i] < ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price re-enters Donchian channel OR breaks above weekly EMA(20)
            if close[i] >= donchian_low[i] or close[i] > ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries with volume confirmation
            if volume[i] > volume_threshold[i]:
                if close[i] > donchian_high[i]:
                    # Breakout above weekly Donchian high - bullish
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donchian_low[i]:
                    # Breakout below weekly Donchian low - bearish
                    signals[i] = -0.25
                    position = -1
    
    return signals
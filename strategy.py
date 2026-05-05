#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Long when price breaks above 6h Donchian(20) upper band AND weekly pivot is bullish (close > weekly PP) AND volume > 1.5x 20-period average
# Short when price breaks below 6h Donchian(20) lower band AND weekly pivot is bearish (close < weekly PP) AND volume > 1.5x 20-period average
# Exit when price retraces to 6h Donchian(20) midpoint OR weekly trend flips
# Uses 6h primary timeframe with 1w HTF for pivot filter to capture institutional flow with low frequency
# Discrete sizing (0.25) to limit fee drag and manage drawdown in both bull and bear markets
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Donchian channels provide objective breakout levels; weekly pivot filter ensures alignment with higher-timeframe structure
# Volume confirmation filters out low-momentum breakouts; works in bull (breakouts continue) and bear (breakdowns accelerate)

name = "6h_Donchian20_Breakout_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for weekly pivot filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot point (PP) from weekly OHLC
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_pp = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly PP to 6h timeframe (using previous week's PP to avoid look-ahead)
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    
    # Calculate 6h Donchian(20) channels
    if len(high) >= 20:
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_upper + donchian_lower) / 2.0
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.5x 20-period average (balanced to avoid overtrading)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(weekly_pp_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND weekly close > weekly PP AND volume spike
            if (high[i] > donchian_upper[i] and 
                close[i] > weekly_pp_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND weekly close < weekly PP AND volume spike
            elif (low[i] < donchian_lower[i] and 
                  close[i] < weekly_pp_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retracement to Donchian midpoint OR weekly trend flips (close < weekly PP)
            if close[i] <= donchian_mid[i] or close[i] < weekly_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retracement to Donchian midpoint OR weekly trend flips (close > weekly PP)
            if close[i] >= donchian_mid[i] or close[i] > weekly_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
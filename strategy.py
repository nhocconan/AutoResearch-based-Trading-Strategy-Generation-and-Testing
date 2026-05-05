#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Width Regime + 1d Donchian Breakout + Volume Spike
# BB Width < 0.03 = low volatility squeeze (regime filter)
# Long when: price breaks above 1d Donchian(20) high AND volume > 2x 20-period MA AND BB Width < 0.03
# Short when: price breaks below 1d Donchian(20) low AND volume > 2x 20-period MA AND BB Width < 0.03
# Exit when: price returns to 6h EMA(20) OR BB Width > 0.06 (volatility expansion)
# Uses volatility regime to filter breakouts, Donchian for structure, volume for confirmation
# Timeframe: 6h, HTF: 1d for Donchian channels. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_BBWidth_Donchian_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands on 6h for volatility regime
    if len(close) >= 20:
        sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
        std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
        upper_bb = sma_20 + (2 * std_20)
        lower_bb = sma_20 - (2 * std_20)
        bb_width = (upper_bb - lower_bb) / sma_20  # normalized width
    else:
        sma_20 = np.full(n, np.nan)
        bb_width = np.full(n, np.nan)
    
    # EMA(20) for exit condition on 6h
    if len(close) >= 20:
        ema_20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    else:
        ema_20 = np.full(n, np.nan)
    
    # Volume confirmation on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # need sufficient data for Donchian
        return np.zeros(n)
    
    # Calculate Donchian(20) channels on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    if len(high_1d) >= 20:
        # Donchian high: highest high of last 20 periods
        donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
        # Donchian low: lowest low of last 20 periods
        donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    else:
        donch_high = np.full(len(df_1d), np.nan)
        donch_low = np.full(len(df_1d), np.nan)
    
    # Align 1d Donchian channels to 6h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bb_width[i]) or np.isnan(ema_20[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price above 1d Donchian high + volume filter + low volatility squeeze
            if (close[i] > donch_high_aligned[i] and 
                volume_filter[i] and 
                bb_width[i] < 0.03):
                signals[i] = 0.25
                position = 1
            # Short conditions: price below 1d Donchian low + volume filter + low volatility squeeze
            elif (close[i] < donch_low_aligned[i] and 
                  volume_filter[i] and 
                  bb_width[i] < 0.03):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to 6h EMA(20) OR volatility expands (BB Width > 0.06)
            if (close[i] <= ema_20[i] or bb_width[i] > 0.06):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to 6h EMA(20) OR volatility expands (BB Width > 0.06)
            if (close[i] >= ema_20[i] or bb_width[i] > 0.06):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
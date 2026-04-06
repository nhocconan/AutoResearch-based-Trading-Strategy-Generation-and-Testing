#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above Donchian(20) high AND 1d ADX > 25 (trending) AND volume > 1.5x average
# Short when price breaks below Donchian(20) low AND 1d ADX > 25 AND volume > 1.5x average
# Exit when price returns to Donchian midpoint or ADX < 20 (range)
# Targets 50-150 total trades over 4 years by requiring multiple confirmations

name = "12h_donchian_adx_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period) on 12h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # ADX (14-period) from 1d timeframe for trend strength
    df_1d = get_htf_data(prices, '1d')
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # True Range
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr3 = np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((daily_high - np.concatenate([[daily_high[0]], daily_high[:-1]])) > 
                       (np.concatenate([[daily_low[0]], daily_low[:-1]]) - daily_low),
                       np.maximum(daily_high - np.concatenate([[daily_high[0]], daily_high[:-1]]), 0), 0)
    dm_minus = np.where((np.concatenate([[daily_low[0]], daily_low[:-1]]) - daily_low) > 
                        (daily_high - np.concatenate([[daily_high[0]], daily_high[:-1]])),
                        np.maximum(np.concatenate([[daily_low[0]], daily_low[:-1]]) - daily_low, 0), 0)
    
    # Smoothed values
    tr_ma = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    dm_plus_ma = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean()
    dm_minus_ma = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean()
    
    # DI and DX
    di_plus = 100 * dm_plus_ma / (tr_ma + 1e-10)
    di_minus = 100 * dm_minus_ma / (tr_ma + 1e-10)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    
    # Align daily ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(adx_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price returns to midpoint OR trend weakens (ADX < 20)
        if position == 1:  # long position
            if close[i] <= donchian_mid[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_mid[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with strong trend and volume
            # Long: break above Donchian high + ADX > 25 + volume confirmation
            if (close[i] > donchian_high[i] and adx_aligned[i] > 25 and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + ADX > 25 + volume confirmation
            elif (close[i] < donchian_low[i] and adx_aligned[i] > 25 and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals
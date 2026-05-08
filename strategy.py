#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with daily volume confirmation and ADX trend filter.
# Long when price breaks above Donchian upper band (20-period high) AND volume > 1.3x 20-period average AND ADX > 25 (trending market).
# Short when price breaks below Donchian lower band (20-period low) AND volume > 1.3x 20-period average AND ADX > 25.
# Exit when price crosses back inside the Donchian channel (between upper and lower bands).
# Uses 4h timeframe with 1d volume and ADX for higher timeframe context.
# Target: 75-200 total trades over 4 years (19-50/year) with controlled frequency to avoid fee drag.

name = "4h_Donchian_20_Volume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for volume and ADX
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period high/low)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily volume filter: current volume > 1.3x 20-period average
    vol_ma20_d = pd.Series(df_d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma20_d_aligned = align_htf_to_ltf(prices, df_d, vol_ma20_d)
    volume_filter = volume > (1.3 * vol_ma20_d_aligned)
    
    # Daily ADX filter (14-period) for trend strength
    # Calculate True Range
    tr1 = df_d['high'] - df_d['low']
    tr2 = np.abs(df_d['high'] - df_d['close'].shift(1))
    tr3 = np.abs(df_d['low'] - df_d['close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr_d = tr.rolling(window=14, min_periods=14).mean()
    
    # Calculate Directional Movement
    up_move = df_d['high'] - df_d['high'].shift(1)
    down_move = df_d['low'].shift(1) - df_d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Calculate Directional Indicators
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).sum() / atr_d)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).sum() / atr_d)
    
    # Calculate ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    adx_aligned = align_htf_to_ltf(prices, df_d, adx_values)
    
    # Trend filter: ADX > 25
    trend_filter = adx_aligned > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(trend_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper, volume filter, trending market
            long_cond = (close[i] > donchian_upper[i]) and volume_filter[i] and trend_filter[i]
            # Short conditions: price breaks below Donchian lower, volume filter, trending market
            short_cond = (close[i] < donchian_lower[i]) and volume_filter[i] and trend_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian lower
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Donchian upper
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
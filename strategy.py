#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with volume confirmation and ADX trend filter.
# Long when price breaks above Donchian upper band (20) AND volume > 2x 20-period average AND ADX > 25.
# Short when price breaks below Donchian lower band (20) AND volume > 2x 20-period average AND ADX > 25.
# Exit when price crosses back inside the Donchian channel.
# Uses 4h timeframe with 1d volume and ADX for higher timeframe context.
# Target: 100-200 total trades over 4 years (25-50/year) with controlled frequency to avoid fee drag.
# Designed to work in both bull and bear markets via trend filter (ADX) and volume confirmation.

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
    
    # Donchian channel (20-period) on 4h data
    donchian_period = 20
    upper = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Align Donchian levels to 4h timeframe (already on 4h, no alignment needed)
    # But we'll keep the structure for consistency
    
    # Daily volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(df_d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma20_aligned = align_htf_to_ltf(prices, df_d, vol_ma20)
    volume_filter = volume > (2.0 * vol_ma20_aligned)
    
    # Daily ADX trend filter (14-period)
    adx_period = 14
    # Calculate True Range
    tr1 = df_d['high'] - df_d['low']
    tr2 = abs(df_d['high'] - df_d['close'].shift(1))
    tr3 = abs(df_d['low'] - df_d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=adx_period, min_periods=adx_period).mean()
    
    # Calculate Directional Movement
    dm_plus = pd.Series(np.where((df_d['high'] - df_d['high'].shift(1)) > (df_d['low'].shift(1) - df_d['low']), 
                                 np.maximum(df_d['high'] - df_d['high'].shift(1), 0), 0))
    dm_minus = pd.Series(np.where((df_d['low'].shift(1) - df_d['low']) > (df_d['high'] - df_d['high'].shift(1)), 
                                  np.maximum(df_d['low'].shift(1) - df_d['low'], 0), 0))
    
    # Smooth DM and TR
    di_plus = 100 * (dm_plus.rolling(window=adx_period, min_periods=adx_period).sum() / atr)
    di_minus = 100 * (dm_minus.rolling(window=adx_period, min_periods=adx_period).sum() / atr)
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=adx_period, min_periods=adx_period).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_d, adx)
    
    # Trend filter: ADX > 25
    trend_filter = adx_aligned > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_period, 20)  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(trend_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper, volume filter, trend filter
            long_cond = (close[i] > upper[i]) and volume_filter[i] and trend_filter[i]
            # Short conditions: price breaks below Donchian lower, volume filter, trend filter
            short_cond = (close[i] < lower[i]) and volume_filter[i] and trend_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian lower
            if close[i] < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Donchian upper
            if close[i] > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
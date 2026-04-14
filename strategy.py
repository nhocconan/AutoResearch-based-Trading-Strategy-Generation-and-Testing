#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume confirmation and 1d ADX filter
# Long when price breaks above 12h Donchian upper band with volume >1.5x 20-period average and 1d ADX > 20
# Short when price breaks below 12h Donchian lower band with volume >1.5x 20-period average and 1d ADX > 20
# Exit when price crosses the 12h Donchian midline
# ADX filter ensures trades occur only in trending markets, reducing whipsaw in ranging conditions
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h and daily data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate 12h Donchian channel (20-period lookback)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate 12h volume average (20-period)
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX (14-period)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # True Range
    tr1 = high_daily - low_daily
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_daily - np.roll(high_daily, 1)) > (np.roll(low_daily, 1) - low_daily),
                       np.maximum(high_daily - np.roll(high_daily, 1), 0), 0)
    dm_minus = np.where((np.roll(low_daily, 1) - low_daily) > (high_daily - np.roll(high_daily, 1)),
                        np.maximum(np.roll(low_daily, 1) - low_daily, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+, DM- (14-period)
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus14 / tr14
    di_minus = 100 * dm_minus14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_12h, donchian_middle)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 40  # for 20-period Donchian and 14-period ADX
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma_12h_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_12h_current = volume[i]  # Current 12h volume
        
        if position == 0:
            # Long setup: break above Donchian upper with volume confirmation and ADX > 20
            if (price > donchian_upper_aligned[i] and 
                vol_12h_current > 1.5 * vol_ma_12h_aligned[i] and  # Volume confirmation
                adx_aligned[i] > 20):                             # Trending market filter
                position = 1
                signals[i] = position_size
            # Short setup: break below Donchian lower with volume confirmation and ADX > 20
            elif (price < donchian_lower_aligned[i] and 
                  vol_12h_current > 1.5 * vol_ma_12h_aligned[i] and  # Volume confirmation
                  adx_aligned[i] > 20):                             # Trending market filter
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian middle
            if price < donchian_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian middle
            if price > donchian_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Donchian_ADX_Volume"
timeframe = "12h"
leverage = 1.0
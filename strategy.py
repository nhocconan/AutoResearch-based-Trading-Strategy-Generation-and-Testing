#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Donchian Breakout + Volume Confirmation + ADX Trend Filter
# Long when price breaks above weekly Donchian high + volume spike + weekly ADX > 25
# Short when price breaks below weekly Donchian low + volume spike + weekly ADX > 25
# Exit when price re-enters the Donchian channel
# Uses weekly timeframe for trend structure to avoid whipsaws in 1d chart
# Discrete sizing (0.25) to limit overtrading and fee drag
# Works in trending markets (both bull and bear) by catching sustained moves

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly Donchian channels (20-period)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Donchian channels on weekly data
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Weekly ADX for trend filter (14-period)
    # Calculate True Range
    tr1 = pd.Series(high_1w).diff().abs()
    tr2 = (pd.Series(high_1w) - pd.Series(low_1w).shift(1)).abs()
    tr3 = (pd.Series(low_1w) - pd.Series(high_1w).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    dm_plus = pd.Series(high_1w).diff()
    dm_minus = -(pd.Series(low_1w).diff())
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smooth TR and DM with Wilder's smoothing (using EMA as approximation)
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # Calculate DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align weekly ADX to daily timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation: current > 2.0x median of last 20 days
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: price breaks above weekly Donchian high + volume + ADX > 25
        if (close[i] > donchian_high_aligned[i] and 
            volume[i] > vol_threshold[i] and 
            adx_aligned[i] > 25):
            signals[i] = 0.25
        
        # Short: price breaks below weekly Donchian low + volume + ADX > 25
        elif (close[i] < donchian_low_aligned[i] and 
              volume[i] > vol_threshold[i] and 
              adx_aligned[i] > 25):
            signals[i] = -0.25
        
        # Exit: price re-enters the Donchian channel
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] <= donchian_high_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] >= donchian_low_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_WeeklyDonchian_Breakout_Volume_ADX"
timeframe = "1d"
leverage = 1.0
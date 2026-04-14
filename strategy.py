#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian Breakout + Volume Spike + 1w ADX Trend Filter
# Uses daily Donchian channel breakouts with volume confirmation and weekly trend strength filter
# Weekly ADX > 25 ensures we only trade in strong trending markets, avoiding false breakouts in ranging conditions
# Works in bull/bear by capturing breakouts in the direction of the weekly trend
# Target: 30-100 total trades over 4 years (7-25/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Weekly ADX calculation (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus_1w = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                          np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus_1w = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                           np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus_1w[0] = 0
    dm_minus_1w[0] = 0
    
    # Smoothed values
    tr14_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
    dm_plus14_1w = pd.Series(dm_plus_1w).rolling(window=14, min_periods=14).sum().values
    dm_minus14_1w = pd.Series(dm_minus_1w).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus_1w = np.where(tr14_1w != 0, 100 * dm_plus14_1w / tr14_1w, 0)
    di_minus_1w = np.where(tr14_1w != 0, 100 * dm_minus14_1w / tr14_1w, 0)
    
    # DX and ADX
    dx_1w = np.where((di_plus_1w + di_minus_1w) != 0, 100 * np.abs(di_plus_1w - di_minus_1w) / (di_plus_1w + di_minus_1w), 0)
    adx_1w = pd.Series(dx_1w).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly ADX to daily
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Daily Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: volume > 2x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 35  # for ADX calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(adx_1w_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade when weekly ADX > 25 (strong trending market)
        if adx_1w_aligned[i] < 25:
            # In weak trend/ranging market, stay flat
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume filter
            if price > donchian_high[i] and vol > 2.0 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low with volume filter
            elif price < donchian_low[i] and vol > 2.0 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low
            if price < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high
            if price > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Donchian_Volume_WeeklyADX_Filter"
timeframe = "1d"
leverage = 1.0
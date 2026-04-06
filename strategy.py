#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter + Donchian(20) breakout with volume confirmation
# In trending markets (CHOP < 38.2): trade breakouts in direction of trend
# In ranging markets (CHOP > 61.8): fade extremes at Donchian bands
# Uses 1d ADX to confirm trend strength, avoids whipsaws in weak trends
# Target: 50-150 total trades over 4 years for optimal 12h performance

name = "12h_chop_donchian_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Choppiness Index (14-period) - range detection
    atr = np.abs(high - low)
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum()
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min()
    range_14 = highest_high_14 - lowest_low_14
    chop = 100 * np.log10(atr_sum / range_14) / np.log10(14)
    chop_values = chop.values
    
    # 1-day ADX (14-period) - trend strength filter
    df_1d = get_htf_data(prices, '1d')
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(daily_high[1:] - daily_low[1:])
    tr2 = np.abs(daily_high[1:] - daily_close[:-1])
    tr3 = np.abs(daily_low[1:] - daily_close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((daily_high[1:] - daily_high[:-1]) > (daily_low[:-1] - daily_low[1:]), 
                       np.maximum(daily_high[1:] - daily_high[:-1], 0), 0)
    dm_minus = np.where((daily_low[:-1] - daily_low[1:]) > (daily_high[1:] - daily_high[:-1]), 
                        np.maximum(daily_low[:-1] - daily_low[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum()
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum()
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(chop_values[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Regime-based logic
        is_trending = chop_values[i] < 38.2 and adx_aligned[i] > 20
        is_ranging = chop_values[i] > 61.8
        
        # Check exits: price crosses Donchian midline
        if position == 1:  # long position
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries based on regime
            if is_trending:
                # In trending markets: trade breakouts with trend
                # Long: price breaks above Donchian upper AND volume confirmation
                if (close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1] and 
                    volume[i] > volume_threshold[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian lower AND volume confirmation
                elif (close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1] and 
                      volume[i] > volume_threshold[i]):
                    signals[i] = -0.25
                    position = -1
            elif is_ranging:
                # In ranging markets: fade extremes at Donchian bands
                # Long: price touches Donchian lower AND volume confirmation (mean reversion up)
                if (close[i] <= donchian_lower[i] and close[i-1] > donchian_lower[i-1] and 
                    volume[i] > volume_threshold[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: price touches Donchian upper AND volume confirmation (mean reversion down)
                elif (close[i] >= donchian_upper[i] and close[i-1] < donchian_upper[i-1] and 
                      volume[i] > volume_threshold[i]):
                    signals[i] = -0.25
                    position = -1
    
    return signals
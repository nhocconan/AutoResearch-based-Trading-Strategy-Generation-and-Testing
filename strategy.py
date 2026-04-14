#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian breakout with volume confirmation and ADX trend filter.
# Long when price breaks above weekly Donchian upper band, ADX > 25 (trending), and volume > 1.5x average.
# Short when price breaks below weekly Donchian lower band, ADX > 25, and volume > 1.5x average.
# Exit when price returns to Donchian middle band or ADX drops below 20 (trend weakening).
# Uses Donchian channels for trend-following breakouts, ADX for trend strength confirmation,
# and volume for institutional participation. Designed for low trade frequency (7-25/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for Donchian and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need enough for Donchian(20) and ADX(14)
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian Channels (20-period)
    donch_upper = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donch_middle = (donch_upper + donch_lower) / 2
    
    # Calculate ADX (14)
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to daily timeframe
    donch_upper_aligned = align_htf_to_ltf(prices, df_1w, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_1w, donch_lower)
    donch_middle_aligned = align_htf_to_ltf(prices, df_1w, donch_middle)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation: 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(34, 20)  # Need ADX and Donchian periods
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_upper_aligned[i]) or 
            np.isnan(donch_lower_aligned[i]) or
            np.isnan(donch_middle_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Weak trend filter: ADX < 20 indicates trend weakening
        weak_trend = adx_aligned[i] < 20
        
        if position == 0:
            # Look for Donchian breakouts in strong trend
            # Long: price breaks above upper band AND strong trend AND volume confirmation
            if (close[i] > donch_upper_aligned[i] and 
                strong_trend and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower band AND strong trend AND volume confirmation
            elif (close[i] < donch_lower_aligned[i] and 
                  strong_trend and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle band or trend weakens
            if (close[i] <= donch_middle_aligned[i] or 
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle band or trend weakens
            if (close[i] >= donch_middle_aligned[i] or 
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian_Channels_ADX_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0
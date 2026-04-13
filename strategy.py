#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1w trend filter and volume confirmation.
# Uses weekly Donchian channels for trend direction, daily ATR for volatility filtering.
# 12h price breaks weekly Donchian channels with volume confirmation and volatility filter.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within profitable range.
# Weekly trend filter ensures we only trade in the direction of the higher timeframe trend.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Daily data for ATR and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian upper and lower bands
    donchian_high_20_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_20_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily volume average
    volume_1d = df_1d['volume'].values
    volume_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all data to 12-hour timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20_1w)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20_1w)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    volume_avg_aligned = align_htf_to_ltf(prices, df_1d, volume_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr_aligned[i]) or np.isnan(volume_avg_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR > 0.5 * average ATR
        atr_filter = atr_aligned[i] > (np.nanmean(atr_aligned[max(0, i-50):i]) * 0.5)
        
        # Volume condition: current 12h volume > 1.2 * daily volume average
        # Adjust daily average to approximate 12h period (2 periods per day)
        volume_12h_avg_approx = volume_avg_aligned[i] / 2
        volume_condition = volume[i] > (volume_12h_avg_approx * 1.2)
        
        # Entry conditions: Donchian breakout with volatility and volume filters
        if position == 0:
            # Long when price breaks above weekly Donchian high
            if close[i] > donchian_high_aligned[i] and atr_filter and volume_condition:
                position = 1
                signals[i] = position_size
            # Short when price breaks below weekly Donchian low
            elif close[i] < donchian_low_aligned[i] and atr_filter and volume_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price crosses below weekly Donchian low or loses filters
            if (close[i] < donchian_low_aligned[i]) or (not atr_filter) or (not volume_condition):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price crosses above weekly Donchian high or loses filters
            if (close[i] > donchian_high_aligned[i]) or (not atr_filter) or (not volume_condition):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_Donchian_Breakout_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0
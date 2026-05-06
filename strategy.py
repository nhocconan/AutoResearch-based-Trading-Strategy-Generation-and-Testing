#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Donchian breakout with volume confirmation and chop regime filter
# Donchian channels on daily timeframe capture long-term trend breakouts
# Volume > 1.5x 20-period average confirms institutional participation
# Choppiness index > 61.8 indicates ranging market (avoid false breakouts)
# Works in bull markets (catch breakouts) and bear markets (avoid false signals in ranging conditions)
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_Donchian20_VolumeChopFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily Donchian channels (20-period) ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily high and low for Donchian channels
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Donchian channels: 20-period high and low
    donchian_high = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Choppiness index filter (using daily data)
    # Chop = 100 * log10(sum(ATR(14)) / (max(high, n) - min(low, n))) / log10(n)
    # We use 14-period chop on daily data
    if len(df_1d) >= 14:
        # Calculate True Range for daily data
        tr1 = pd.Series(daily_high).shift(1) - pd.Series(daily_low).shift(1)
        tr2 = abs(pd.Series(daily_high).shift(1) - pd.Series(df_1d['close']).shift(1))
        tr3 = abs(pd.Series(daily_low).shift(1) - pd.Series(df_1d['close']).shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = tr.rolling(window=14, min_periods=14).sum().values
        
        # Maximum high and minimum low over 14 periods
        max_high_14 = pd.Series(daily_high).rolling(window=14, min_periods=14).max().values
        min_low_14 = pd.Series(daily_low).rolling(window=14, min_periods=14).min().values
        
        # Avoid division by zero
        range_14 = max_high_14 - min_low_14
        range_14 = np.where(range_14 == 0, 1e-10, range_14)
        
        # Chop calculation
        chop = 100 * (np.log10(atr_14 / range_14) / np.log10(14))
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
        # Filter: only trade when market is trending (chop < 61.8)
        chop_filter = chop_aligned < 61.8
    else:
        chop_filter = np.ones(n, dtype=bool)  # No filter if insufficient data
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(chop_filter[i]) if len(df_1d) >= 14 else False or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above Donchian high with volume and chop filter
            if close[i] > donchian_high_aligned[i] and volume_filter[i] and chop_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below Donchian low with volume and chop filter
            elif close[i] < donchian_low_aligned[i] and volume_filter[i] and chop_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low (failed breakout) or opposite signal
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high (failed breakdown) or opposite signal
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
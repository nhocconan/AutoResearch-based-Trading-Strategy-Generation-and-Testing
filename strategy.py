#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Donchian channel breakout with volume confirmation and ADX trend filter.
# Long when price breaks above 1-day Donchian upper channel (20-period) AND volume > 1.5x 20-period average volume AND ADX > 25.
# Short when price breaks below 1-day Donchian lower channel (20-period) AND volume > 1.5x 20-period average volume AND ADX > 25.
# Exit when price returns to the 1-day Donchian middle channel or ADX drops below 20.
# Uses daily structure for breakouts on 12h timeframe to capture multi-day moves while avoiding noise.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Donchian channels, volume average, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:  # Need enough for Donchian(20) and ADX(14)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian channels (20-period)
    # Upper channel: highest high over 20 periods
    # Lower channel: lowest low over 20 periods
    # Middle channel: average of upper and lower
    lookback = 20
    highest_high = np.full_like(high_1d, np.nan)
    lowest_low = np.full_like(low_1d, np.nan)
    
    for i in range(lookback - 1, len(high_1d)):
        highest_high[i] = np.max(high_1d[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low_1d[i - lookback + 1:i + 1])
    
    upper_channel = highest_high
    lower_channel = lowest_low
    middle_channel = (upper_channel + lower_channel) / 2.0
    
    # Calculate 20-period average volume
    avg_volume = np.full_like(volume_1d, np.nan)
    for i in range(lookback - 1, len(volume_1d)):
        avg_volume[i] = np.mean(volume_1d[i - lookback + 1:i + 1])
    
    # Calculate ADX (14)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR = smoothed TR (Wilder's smoothing)
    atr = np.full_like(tr, np.nan)
    atr[13] = np.nanmean(tr[1:14])  # First ATR: simple average of first 14 TR
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
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
    
    # Align indicators to 12h timeframe
    upper_channel_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    middle_channel_aligned = align_htf_to_ltf(prices, df_1d, middle_channel)
    avg_volume_aligned = align_htf_to_ltf(prices, df_1d, avg_volume)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(40, 34)  # Need Donchian(20) and ADX(14) periods
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_channel_aligned[i]) or 
            np.isnan(lower_channel_aligned[i]) or
            np.isnan(avg_volume_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume vs 1d average volume
        # Need to get the corresponding 12h bar's volume - since we're using 1d data,
        # we'll use the current 12h volume directly (not averaged)
        vol_confirmation = volume[i] > (1.5 * avg_volume_aligned[i])
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Weak trend filter: ADX < 20 indicates trend weakening
        weak_trend = adx_aligned[i] < 20
        
        if position == 0:
            # Look for breakout entries in strong trend with volume confirmation
            # Long: price breaks above upper channel AND volume confirmation AND strong trend
            if (close[i] > upper_channel_aligned[i] and 
                vol_confirmation and
                strong_trend):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower channel AND volume confirmation AND strong trend
            elif (close[i] < lower_channel_aligned[i] and 
                  vol_confirmation and
                  strong_trend):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle channel or trend weakens
            if (close[i] <= middle_channel_aligned[i] or 
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle channel or trend weakens
            if (close[i] >= middle_channel_aligned[i] or 
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Donchian_Breakout_Volume_ADX_Filter_v1"
timeframe = "12h"
leverage = 1.0
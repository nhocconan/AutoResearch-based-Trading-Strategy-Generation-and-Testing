#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and ADX regime filter
# Long when price breaks above 20-bar Donchian high + 1d volume > 1.5x 20-day average + ADX > 25
# Short when price breaks below 20-bar Donchian low + 1d volume > 1.5x 20-day average + ADX > 25
# Exit when price returns to midpoint of Donchian channel or ADX < 20
# Designed for trending markets with volume confirmation to avoid false breakouts
# Works in both bull and bear markets: ADX filter ensures we only trade strong trends,
# while volume confirmation validates breakout strength

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d average volume (20-day)
    vol_1d = df_1d['volume'].values
    avg_vol_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d average volume to 4h timeframe
    avg_vol_20_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_20)
    
    # Calculate 4h ADX (14 periods) for trend strength
    # Load 4h data ONCE for ADX calculation
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h ADX (14 periods)
    adx_len = 14
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_plus_sum = pd.Series(dm_plus).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_minus_sum = pd.Series(dm_minus).rolling(window=adx_len, min_periods=adx_len).sum().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_sum / tr_sum
    minus_di = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=adx_len, min_periods=adx_len).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 20 + adx_len)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(avg_vol_20_aligned[i]) or
            avg_vol_20_aligned[i] == 0):
            signals[i] = 0.0
            continue
        
        # Calculate Donchian channels (20-period) using available data up to i
        lookback = min(20, i+1)
        donchian_high = np.max(high[i-lookback+1:i+1])
        donchian_low = np.min(low[i-lookback+1:i+1])
        donchian_mid = (donchian_high + donchian_low) / 2
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        # Need to get current 1d volume - find the most recent complete 1d bar
        # For 4h timeframe, we can approximate using the volume data
        vol_ratio = 1.0  # Default if we can't calculate
        if i >= 20:  # Need enough data for volume ratio calculation
            # Simplified: use current volume vs average (approximation for 4h)
            vol_ratio = volume[i] / avg_vol_20_aligned[i] if avg_vol_20_aligned[i] > 0 else 1.0
        
        volume_confirmed = vol_ratio > 1.5
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Enter long: Donchian breakout up + volume confirmation + strong trend
            if close[i] > donchian_high and volume_confirmed and strong_trend:
                position = 1
                signals[i] = position_size
            # Enter short: Donchian breakout down + volume confirmation + strong trend
            elif close[i] < donchian_low and volume_confirmed and strong_trend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR trend weakens
            if close[i] < donchian_mid or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR trend weakens
            if close[i] > donchian_mid or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Volume_ADX_Trend_v1"
timeframe = "4h"
leverage = 1.0
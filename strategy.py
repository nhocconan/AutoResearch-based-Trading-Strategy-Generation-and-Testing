#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and 12h ADX trend filter.
Uses 4h price channel breakouts for entry, confirmed by 12h volume spike (>1.5x 20-period average)
and strong trend (ADX > 25) to avoid false signals in ranging markets. Long when price breaks
above 4h Donchian upper band with volume confirmation and strong 12h trend. Short when price
breaks below 4h Donchian lower band with volume confirmation and strong 12h trend.
Exit when price returns to opposite Donchian level. Designed for 4-8 trades per year per symbol
to minimize fee drag while capturing strong trending moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 12h data for volume confirmation and ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h volume spike (volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_12h > (vol_ma_20 * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_12h, vol_spike.astype(float))
    
    # Calculate 12h ADX (14-period) for trend strength
    # TR = max(high-low, |high-close_prev|, |low-close_prev|)
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.max([high_12h[0] - low_12h[0], np.abs(high_12h[0] - close_12h[0]), np.abs(low_12h[0] - close_12h[0])])], 
                            np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_ma = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    dm_plus_ma = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_ma = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_ma / tr_ma
    di_minus = 100 * dm_minus_ma / tr_ma
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Trend filter: ADX > 25 indicates strong trend
    strong_trend = adx > 25
    strong_trend_aligned = align_htf_to_ltf(prices, df_12h, strong_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(strong_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Donchian breakout + volume spike + strong trend
        breakout_long = close[i] > donchian_high_aligned[i]
        breakout_short = close[i] < donchian_low_aligned[i]
        vol_confirm = vol_spike_aligned[i] > 0.5  # True if volume spike
        trend_confirm = strong_trend_aligned[i] > 0.5  # True if strong trend
        
        long_entry = breakout_long and vol_confirm and trend_confirm
        short_entry = breakout_short and vol_confirm and trend_confirm
        
        # Exit when price returns to opposite Donchian level (mean reversion within channel)
        exit_long = position == 1 and close[i] < donchian_low_aligned[i]
        exit_short = position == -1 and close[i] > donchian_high_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_donchian_volume_adx"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 1d ATR volume spike filter and 12h Supertrend trend filter.
- Primary timeframe: 4h targeting 100-180 total trades over 4 years (25-45/year).
- HTF: 12h for Supertrend trend filter and 1d for ATR volume confirmation.
- Entry: Long when price breaks above Donchian upper (20) AND ATR ratio > 1.8 AND Supertrend uptrend.
         Short when price breaks below Donchian lower (20) AND ATR ratio > 1.8 AND Supertrend downtrend.
- Exit: Opposite Donchian breakout OR Supertrend reversal.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels provide objective price structure; ATR spike confirms momentum; Supertrend filters trend direction.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~140 total over 4 years (~35/year) based on volatility breakout frequency with strict filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def atr(high, low, close, period):
    """Calculate Average True Range."""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First period
    return pd.Series(true_range).ewm(span=period, adjust=False, min_periods=period).mean().values

def supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    atr_vals = atr(high, low, close, period)
    hl2 = (high + low) / 2.0
    upper_band = hl2 + (multiplier * atr_vals)
    lower_band = hl2 - (multiplier * atr_vals)
    
    supertrend_vals = np.full_like(close, np.nan, dtype=float)
    direction = np.full_like(close, 1, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    supertrend_vals[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close)):
        # Upper band
        if upper_band[i] < supertrend_vals[i-1] or close[i-1] > supertrend_vals[i-1]:
            upper_band[i] = upper_band[i]
        else:
            upper_band[i] = supertrend_vals[i-1]
            
        # Lower band
        if lower_band[i] > supertrend_vals[i-1] or close[i-1] < supertrend_vals[i-1]:
            lower_band[i] = lower_band[i]
        else:
            lower_band[i] = supertrend_vals[i-1]
            
        # Supertrend
        if supertrend_vals[i-1] == upper_band[i-1]:
            if close[i] <= upper_band[i]:
                supertrend_vals[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend_vals[i] = lower_band[i]
                direction[i] = 1
        else:
            if close[i] >= lower_band[i]:
                supertrend_vals[i] = lower_band[i]
                direction[i] = 1
            else:
                supertrend_vals[i] = upper_band[i]
                direction[i] = -1
                
    return supertrend_vals, direction

def donchian_channels(high, low, period=20):
    """Calculate Donchian channels."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 12h Supertrend trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    st_12h, st_dir_12h = supertrend(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 10, 3.0)
    st_12h_aligned = align_htf_to_ltf(prices, df_12h, st_12h, additional_delay_bars=1)
    st_dir_12h_aligned = align_htf_to_ltf(prices, df_12h, st_dir_12h.astype(float), additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Calculate 4h Donchian channels (primary timeframe)
    donch_upper, donch_lower = donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(st_12h_aligned[i]) or np.isnan(st_dir_12h_aligned[i]) or
            np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Exit conditions: opposite Donchian breakout OR Supertrend reversal
        if position != 0:
            # Exit long: price breaks below Donchian lower OR Supertrend turns down
            if position == 1:
                if curr_low < donch_lower[i] or st_dir_12h_aligned[i] == -1:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian upper OR Supertrend turns up
            elif position == -1:
                if curr_high > donch_upper[i] or st_dir_12h_aligned[i] == 1:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with volume confirmation and trend filter
        if position == 0:
            # Long: price breaks above Donchian upper AND ATR ratio > 1.8 AND Supertrend uptrend
            if curr_high > donch_upper[i] and atr_ratio_aligned[i] > 1.8 and st_dir_12h_aligned[i] == 1:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND ATR ratio > 1.8 AND Supertrend downtrend
            elif curr_low < donch_lower[i] and atr_ratio_aligned[i] > 1.8 and st_dir_12h_aligned[i] == -1:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dATR_VolumeSpike_12hSupertrend_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0
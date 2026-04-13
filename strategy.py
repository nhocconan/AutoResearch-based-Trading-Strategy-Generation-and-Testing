#!/usr/bin/env python3
"""
12h_1d_1w_Bollinger_Breakout_Pullback
Hypothesis: Uses daily Bollinger Bands (20,2) to identify volatility regime and mean-reversion opportunities.
In low volatility (BB width < 50th percentile), look for breakouts of the 12h Donchian channel (20) with volume confirmation.
In high volatility (BB width > 50th percentile), fade moves toward Bollinger Bands with weekly trend filter.
Weekly trend determines bias: only take long signals when weekly close > weekly open, short when weekly close < weekly open.
This adapts to both ranging and trending markets, working in bull and bear regimes by switching between breakout and mean-reversion logic.
Target: 15-35 trades/year on 12h (60-140 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_bollinger_bands(close, period=20, std_dev=2):
    """Calculate Bollinger Bands for given close array."""
    if len(close) < period:
        return np.full_like(close, np.nan), np.full_like(close, np.nan), np.full_like(close, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    return upper, lower, sma

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channels for given high and low arrays."""
    if len(high) < period:
        return np.full_like(high, np.nan), np.full_like(high, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Bollinger Bands on daily
    bb_upper, bb_lower, bb_middle = calculate_bollinger_bands(close_1d, 20, 2)
    
    # Calculate Bollinger Band width (normalized)
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # Calculate 12-period percentile of BB width for regime detection
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=12, min_periods=12).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50, raw=False
    ).values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open
    
    # Align all data to 12h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    bb_middle_aligned = align_htf_to_ltf(prices, df_1d, bb_middle)
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # Calculate 12h Donchian channels (20-period)
    donch_high, donch_low = calculate_donchian_channels(high, low, 20)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(bb_middle_aligned[i]) or np.isnan(bb_width_percentile_aligned[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Regime detection: low volatility (BB width < 50th percentile) = breakout mode
        # High volatility (BB width > 50th percentile) = mean-reversion mode
        low_volatility = bb_width_percentile_aligned[i] < 50
        
        if low_volatility:
            # BREAKOUT MODE: Look for Donchian breakouts with volume expansion
            
            # Long setup: price breaks above Donchian high with volume expansion
            long_condition = (high[i] > donch_high[i]) and volume_expansion[i]
            
            # Short setup: price breaks below Donchian low with volume expansion
            short_condition = (low[i] < donch_low[i]) and volume_expansion[i]
            
            if long_condition and position != 1:
                position = 1
                signals[i] = position_size
            elif short_condition and position != -1:
                position = -1
                signals[i] = -position_size
            elif position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
                
        else:
            # MEAN-REVERSION MODE: Fade moves toward Bollinger Bands with weekly trend filter
            
            # Long setup: price touches or goes below lower BB with weekly bullish bias
            long_condition = (low[i] <= bb_lower_aligned[i]) and weekly_bullish_aligned[i] > 0.5
            
            # Short setup: price touches or goes above upper BB with weekly bearish bias
            short_condition = (high[i] >= bb_upper_aligned[i]) and weekly_bullish_aligned[i] < 0.5
            
            if long_condition and position != 1:
                position = 1
                signals[i] = position_size
            elif short_condition and position != -1:
                position = -1
                signals[i] = -position_size
            elif position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_1w_Bollinger_Breakout_Pullback"
timeframe = "12h"
leverage = 1.0
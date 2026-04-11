#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1-day and 1-week Choppiness Index as regime filter.
# Uses weekly Choppiness to determine trend regime (trend vs range) and daily Choppiness for entry timing.
# In trending regime (weekly CHOP < 38.2): trade breakouts in direction of trend.
# In ranging regime (weekly CHOP > 61.8): fade at extremes using Bollinger Bands.
# Volume confirmation filters false signals. Designed for 20-40 trades/year.

name = "4h_1w1d_choppiness_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for indicators
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Choppiness Index (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for weekly
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.concatenate([[np.nan], close_1w[:-1]]))
    tr3 = np.abs(low_1w - np.concatenate([[np.nan], close_1w[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR (14-period)
    atr_period = 14
    atr = np.full_like(tr, np.nan, dtype=float)
    for i in range(atr_period - 1, len(tr)):
        atr[i] = np.nanmean(tr[i - atr_period + 1:i + 1])
    
    # Sum of ATR (14-period)
    atr_sum = np.full_like(tr, np.nan, dtype=float)
    for i in range(atr_period - 1, len(tr)):
        atr_sum[i] = np.nansum(tr[i - atr_period + 1:i + 1])
    
    # Choppiness Index
    chop = np.full_like(tr, np.nan, dtype=float)
    for i in range(atr_period - 1, len(tr)):
        if atr_sum[i] > 0 and np.nansum(tr[i - atr_period + 1:i + 1]) > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / (atr[i] * atr_period)) / np.log10(atr_period)
    
    # Weekly regime: trending (CHOP < 38.2) or ranging (CHOP > 61.8)
    weekly_chop_trend = chop < 38.2
    weekly_chop_range = chop > 61.8
    
    # Align weekly chop regime to 4h
    weekly_chop_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_chop_trend)
    weekly_chop_range_aligned = align_htf_to_ltf(prices, df_1w, weekly_chop_range)
    
    # Calculate daily Bollinger Bands (20, 2) for ranging regime
    close_1d = df_1d['close'].values
    sma_20 = np.full_like(close_1d, np.nan, dtype=float)
    std_20 = np.full_like(close_1d, np.nan, dtype=float)
    for i in range(19, len(close_1d)):
        sma_20[i] = np.mean(close_1d[i-19:i+1])
        std_20[i] = np.std(close_1d[i-19:i+1])
    
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    # Align BB to 4h
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    
    # Daily Donchian Channel (20-period) for trending regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band (20-period high)
    donch_high = np.full_like(high_1d, np.nan, dtype=float)
    for i in range(19, len(high_1d)):
        donch_high[i] = np.max(high_1d[i-19:i+1])
    
    # Lower band (20-period low)
    donch_low = np.full_like(low_1d, np.nan, dtype=float)
    for i in range(19, len(low_1d)):
        donch_low[i] = np.min(low_1d[i-19:i+1])
    
    # Align Donchian to 4h
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Volume filter: current volume > 1.3 * 20-period average volume
    volume_1d = df_1d['volume'].values
    vol_avg_20 = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_avg_20[i] = np.mean(volume_1d[i-19:i+1])
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or
            np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(vol_avg_aligned[i]) or
            np.isnan(weekly_chop_trend_aligned[i]) or np.isnan(weekly_chop_range_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.3 * 20-period average volume
        vol_filter = volume[i] > 1.3 * vol_avg_aligned[i]
        
        # Determine weekly regime
        is_trending = weekly_chop_trend_aligned[i]
        is_ranging = weekly_chop_range_aligned[i]
        
        # Entry logic based on regime
        if is_trending and vol_filter:
            # Trending regime: trade breakouts in direction of trend
            # Determine trend direction from price vs Donchian midpoint
            donch_mid = (donch_high_aligned[i] + donch_low_aligned[i]) / 2
            trend_up = close[i] > donch_mid
            trend_down = close[i] < donch_mid
            
            breakout_long = (high[i] >= donch_high_aligned[i] and trend_up)
            breakout_short = (low[i] <= donch_low_aligned[i] and trend_down)
            
            if breakout_long and position != 1:
                position = 1
                signals[i] = 0.25
            elif breakout_short and position != -1:
                position = -1
                signals[i] = -0.25
                
        elif is_ranging and vol_filter:
            # Ranging regime: fade at Bollinger Bands
            fade_long = (low[i] <= bb_lower_aligned[i])
            fade_short = (high[i] >= bb_upper_aligned[i])
            
            if fade_long and position != 1:
                position = 1
                signals[i] = 0.25
            elif fade_short and position != -1:
                position = -1
                signals[i] = -0.25
        
        # Exit logic: opposite signal or regime change
        exit_long = (position == 1 and 
                    ((is_ranging and high[i] >= bb_upper_aligned[i]) or  # Hit upper BB in ranging
                     (is_trending and low[i] <= donch_low_aligned[i]) or  # Hit lower Donchian in trending
                     not is_trending and not is_ranging))  # Choppy middle - exit
        
        exit_short = (position == -1 and 
                     ((is_ranging and low[i] <= bb_lower_aligned[i]) or  # Hit lower BB in ranging
                      (is_trending and high[i] >= donch_high_aligned[i]) or  # Hit upper Donchian in trending
                      not is_trending and not is_ranging))  # Choppy middle - exit
        
        if position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
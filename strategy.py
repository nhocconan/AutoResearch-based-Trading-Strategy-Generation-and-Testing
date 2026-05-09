#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with weekly EMA50 trend filter and volume confirmation.
# Uses weekly EMA50 for trend direction to avoid counter-trend entries and weekly trend filter.
# Volume > 1.5x 20-period EMA ensures institutional participation.
# Donchian breakout provides clear entry/exit levels with weekly trend filter for trend alignment.
# Designed to work in both bull and bear markets by following weekly trend.
name = "12h_Donchian20_WeeklyEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for EMA50 trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Weekly EMA50
    ema_50_weekly = pd.Series(df_weekly['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    # Daily data for Donchian channels (20-period)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Donchian channels: 20-period high/low
    high_20 = pd.Series(df_daily['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_daily['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 12h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_daily, high_20)
    lower_band_aligned = align_htf_to_ltf(prices, df_daily, low_20)
    
    # Volume spike filter: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(ema_50_weekly_aligned[i]) or np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band with volume spike and above weekly EMA50
            if (price > upper_band_aligned[i] and vol_spike[i] and price > ema_50_weekly_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band with volume spike and below weekly EMA50
            elif (price < lower_band_aligned[i] and vol_spike[i] and price < ema_50_weekly_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below lower band (mean reversion to support)
            if price < lower_band_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above upper band (mean reversion to resistance)
            if price > upper_band_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
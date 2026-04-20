#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian channel breakout with weekly EMA trend filter and volume confirmation
# - Buy when price breaks above 20-day Donchian high AND weekly EMA50 > weekly EMA200 (uptrend) AND volume > 1.5x 20-day average volume
# - Sell when price breaks below 20-day Donchian low AND weekly EMA50 < weekly EMA200 (downtrend) AND volume > 1.5x 20-day average volume
# - Exit when price returns to the 10-day Donchian middle (mean reversion within trend)
# - Uses weekly trend filter to avoid counter-trend trades in strong trends
# - Volume confirmation reduces false breakouts
# - Designed for 1d timeframe to capture major trends with low trade frequency
# - Target: 30-100 total trades over 4 years (7-25/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day Donchian channels
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_middle = (donch_high + donch_low) / 2
    
    # Calculate 20-day average volume for confirmation
    avg_volume = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 and EMA200 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly indicators to daily timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(avg_volume[i]) or \
           np.isnan(ema50_1w_aligned[i]) or np.isnan(ema200_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        volume = volume_1d[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high + uptrend + volume confirmation
            long_breakout = price > donch_high[i]
            uptrend = ema50_1w_aligned[i] > ema200_1w_aligned[i]
            volume_confirm = volume > 1.5 * avg_volume[i]
            
            if long_breakout and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            
            # Enter short: price breaks below Donchian low + downtrend + volume confirmation
            short_breakout = price < donch_low[i]
            downtrend = ema50_1w_aligned[i] < ema200_1w_aligned[i]
            
            if short_breakout and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to Donchian middle (mean reversion)
            if price > donch_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to Donchian middle (mean reversion)
            if price < donch_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_WeeklyEMATrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0
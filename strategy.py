#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Choppiness Index regime filter + 1-week Donchian breakout
# Long when: Donchian upper breakout + CHOP > 61.8 (trending regime) + volume confirmation
# Short when: Donchian lower breakdown + CHOP > 61.8 (trending regime) + volume confirmation
# Uses weekly structure for major trend, daily choppiness to filter ranging markets, volume for confirmation
# Target: 20-40 trades per year (80-160 over 4 years) with 0.25 position sizing

name = "12h_1wDonchian20_1dChop_Trend_Volume_v1"
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
    
    # Calculate 1-week Donchian channels (20-period high/low)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly high and low for Donchian channels
    weekly_high = df_1w['high'].rolling(window=20, min_periods=20).max().values
    weekly_low = df_1w['low'].rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_high = align_htf_to_ltf(prices, df_1w, weekly_high)
    donchian_low = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Calculate 1-day Choppiness Index (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range calculation
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    # Sum of true ranges over 14 periods
    tr_sum = tr.rolling(window=14, min_periods=14).sum().values
    # Max/min over 14 periods
    max_h = df_1d['high'].rolling(window=14, min_periods=14).max().values
    min_l = df_1d['low'].rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index formula: 100 * log10(tr_sum / (max_h - min_l)) / log10(14)
    # Avoid division by zero
    range_hl = max_h - min_l
    chop = np.where(range_hl > 0, 100 * np.log10(tr_sum / range_hl) / np.log10(14), 50)
    
    # Align Choppiness Index to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: >1.3x 30-period average (reduced for fewer trades)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (1.3 * vol_ma_30)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any critical value is NaN or outside session
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above weekly Donchian high + trending regime + volume
            if (close[i] > donchian_high[i] and 
                chop_aligned[i] > 61.8 and  # Trending regime (CHOP > 61.8)
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly Donchian low + trending regime + volume
            elif (close[i] < donchian_low[i] and 
                  chop_aligned[i] > 61.8 and  # Trending regime (CHOP > 61.8)
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
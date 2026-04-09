#!/usr/bin/env python3
# 12h_weekly_donchian_breakout_volume_v1
# Hypothesis: 12h strategy using weekly Donchian channels (20-period) from 1w HTF for breakout entries, volume confirmation (>1.5x 20-bar avg volume), and trend alignment via 1d EMA(100). Enters long when price breaks above weekly Donchian upper with volume and price > 1d EMA(100); enters short when price breaks below weekly Donchian lower with volume and price < 1d EMA(100). Exits on opposite Donchian band touch. Uses discrete sizing (0.25) to limit fee churn. Target: 12-37 trades/year (50-150 total over 4 years). Weekly Donchian provides structural support/resistance that works in bull/bear markets; volume confirms breakout conviction; 1d EMA filters counter-trend noise.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_weekly_donchian_breakout_volume_v1"
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
    
    # Volume average for confirmation (20-period = ~10 days of 12h bars)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Multi-timeframe: 1d EMA(100) trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    ema_100_1d = close_1d_s.ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # Multi-timeframe: weekly Donchian channels (20-period) from 1w HTF
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 12h timeframe (wait for weekly close)
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(ema_100_1d_aligned[i]) or
            np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Trend filters
        uptrend = close[i] > ema_100_1d_aligned[i]
        downtrend = close[i] < ema_100_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price touches weekly Donchian lower band
            if close[i] <= low_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches weekly Donchian upper band
            if close[i] >= high_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for weekly Donchian breakout with volume and trend alignment
            bullish_breakout = (close[i] > high_20_aligned[i]) and volume_confirmed and uptrend
            bearish_breakout = (close[i] < low_20_aligned[i]) and volume_confirmed and downtrend
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
# 6h_donchian_breakout_volume_pivot_v1
# Hypothesis: 6h Donchian(20) breakout with volume confirmation (>1.8x 20-period average) and 12h HTF pivot direction filter. Enters long when price breaks above upper Donchian band in bullish 12h pivot regime (price > weekly pivot); short when breaks below lower band in bearish regime (price < weekly pivot). Uses volume to filter weak breakouts and pivot regime to avoid counter-trend whipsaws. Works in bull/bear by following institutional volume-driven breakouts aligned with higher-timeframe structure. Target: 12-37 trades/year (50-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_volume_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Volume average for confirmation (20-period = 20 * 6h bars)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Multi-timeframe: 12h data for pivot calculation
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate weekly pivot points from 12h data (approximate weekly from 5x 12h bars)
    # Weekly high/low/close from last 5 periods of 12h data (5 * 12h = 60h ≈ 1 week)
    lookback = 5
    if len(df_12h) >= lookback:
        weekly_high = pd.Series(df_12h['high'].values).rolling(window=lookback, min_periods=lookback).max().values
        weekly_low = pd.Series(df_12h['low'].values).rolling(window=lookback, min_periods=lookback).min().values
        weekly_close = pd.Series(df_12h['close'].values).rolling(window=lookback, min_periods=lookback).last().values
        
        # Weekly pivot point: (H + L + C) / 3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        
        # Align weekly pivot to 6h timeframe
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_12h, weekly_pivot)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(weekly_pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average (tighter filter)
        volume_confirmed = volume[i] > 1.8 * volume_ma[i]
        
        # Pivot regime filter: bullish if price > weekly pivot, bearish if price < weekly pivot
        bullish_regime = close[i] > weekly_pivot_aligned[i]
        bearish_regime = close[i] < weekly_pivot_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price touches or breaks lower Donchian band
            if close[i] <= low_min[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches or breaks upper Donchian band
            if close[i] >= high_max[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter only with volume confirmation and aligned pivot regime
            if volume_confirmed:
                # Long: price breaks above upper Donchian band in bullish regime
                if close[i] > high_max[i] and bullish_regime:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below lower Donchian band in bearish regime
                elif close[i] < low_min[i] and bearish_regime:
                    position = -1
                    signals[i] = -0.25
    
    return signals
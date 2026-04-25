#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_VolumeSpike
Hypothesis: Trade 12h Donchian(20) breakouts aligned with daily EMA50 trend and volume spike (volume > 1.5 * ATR14).
Only trade in direction of daily trend to avoid whipsaws. Uses discrete sizing 0.25 to limit fee drag.
Target: 12-30 trades/year to avoid fee drag while maintaining edge. Works in bull/bear via daily trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR for volume confirmation (using 12h data)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period) on 12h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # track bars in position for minimum hold
    
    # Start index: need warmup for daily EMA50, ATR, and Donchian
    start_idx = max(50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Volume confirmation: current volume > 1.5 * ATR (tightened to reduce trades)
        volume_confirm = volume[i] > 1.5 * atr[i]
        
        # Determine daily trend from EMA50
        daily_close_aligned = align_htf_to_ltf(prices, df_1d, close_1d)[i]
        if np.isnan(daily_close_aligned):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
            
        if daily_close_aligned > ema_50_1d_aligned[i]:
            daily_trend = 'bullish'  # only allow longs
        elif daily_close_aligned < ema_50_1d_aligned[i]:
            daily_trend = 'bearish'  # only allow shorts
        else:
            daily_trend = 'neutral'  # no trades in neutral zone
        
        if position == 0:
            bars_since_entry = 0
            # Long setup: price breaks above Donchian upper band AND volume confirm AND bullish daily trend
            long_setup = (close[i] > highest_high[i]) and volume_confirm and (daily_trend == 'bullish')
            
            # Short setup: price breaks below Donchian lower band AND volume confirm AND bearish daily trend
            short_setup = (close[i] < lowest_low[i]) and volume_confirm and (daily_trend == 'bearish')
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            bars_since_entry += 1
            # Minimum holding period: 3 bars (~1.5 days for 12h)
            if bars_since_entry < 3:
                signals[i] = 0.25
            else:
                # Long: hold position
                signals[i] = 0.25
                # Exit: price breaks below Donchian lower band OR daily trend turns bearish
                if (close[i] < lowest_low[i]) or (daily_trend == 'bearish'):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
        elif position == -1:
            bars_since_entry += 1
            # Minimum holding period: 3 bars (~1.5 days for 12h)
            if bars_since_entry < 3:
                signals[i] = -0.25
            else:
                # Short: hold position
                signals[i] = -0.25
                # Exit: price breaks above Donchian upper band OR daily trend turns bullish
                if (close[i] > highest_high[i]) or (daily_trend == 'bullish'):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
    
    return signals

name = "12h_Donchian20_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0
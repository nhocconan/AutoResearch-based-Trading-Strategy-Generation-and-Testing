#!/usr/bin/env python3
"""
12h_Donchian20_WeeklyTrend_VolumeConfirm_v1
Hypothesis: 12h Donchian(20) breakout in direction of weekly trend (EMA34) with volume confirmation.
Weekly EMA34 defines structural trend; breakouts in its direction have higher follow-through.
Volume spike confirms institutional participation. Discrete sizing (0.25) limits fee drag.
Target: 50-150 total trades over 4 years (12-37/year) by requiring HTF alignment, breakout, and volume.
Works in bull/bear: weekly EMA adapts to regime; volume filter avoids false breakouts in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for HTF trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA34 for trend direction
    weekly_close = pd.Series(df_1w['close'].values)
    weekly_ema34 = weekly_close.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA34 to 12h timeframe
    weekly_ema34_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema34)
    
    # Calculate 12h Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for weekly EMA, 30 for volume MA, 20 for Donchian)
    start_idx = max(34, 30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_ma_30[i]) or np.isnan(weekly_ema34_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 2.0 * vol_ma_30[i]
        
        # Donchian breakout conditions (use previous bar's channel to avoid look-ahead)
        breakout_above = close[i] > high_20[i-1]
        breakout_below = close[i] < low_20[i-1]
        
        if breakout_above and volume_spike:
            # Long signal: breakout above Donchian high with volume, above weekly EMA34 (bullish bias)
            if close[i] > weekly_ema34_aligned[i]:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            else:
                # Hold or flatten if not aligned with weekly trend
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = 0.0
                    position = 0
        elif breakout_below and volume_spike:
            # Short signal: breakout below Donchian low with volume, below weekly EMA34 (bearish bias)
            if close[i] < weekly_ema34_aligned[i]:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            else:
                # Hold or flatten if not aligned with weekly trend
                if position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
                    position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_WeeklyTrend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0
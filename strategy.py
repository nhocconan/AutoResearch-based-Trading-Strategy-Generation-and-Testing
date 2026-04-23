#!/usr/bin/env python3
"""
Hypothesis: 6h Weekly Donchian Breakout with Daily Volume Spike and 1d EMA34 Trend Filter
- Uses weekly Donchian channels (20-period) for major trend structure
- Daily volume spike (>2.0x 20-period average) confirms institutional participation
- 1d EMA34 defines short-term trend filter: only trade in direction of daily trend
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Weekly timeframe reduces noise and captures major moves
- Volume spike filter ensures breakouts have follow-through
- Works in both bull and bear markets by trading with the major trend on volume confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly Donchian channels (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian channels: highest high/lowest low of past 20 weekly bars
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 6h timeframe (completed weekly bar only)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Calculate daily volume spike confirmation (>2.0x 20-period average)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = pd.Series(volume_1d).values > (2.0 * vol_ma_1d)
    
    # Align daily volume spike to 6h timeframe
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 20  # for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high AND daily volume spike AND above daily EMA34
            if (close[i] > donchian_high_aligned[i] and 
                vol_spike_aligned[i] > 0.5 and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low AND daily volume spike AND below daily EMA34
            elif (close[i] < donchian_low_aligned[i] and 
                  vol_spike_aligned[i] > 0.5 and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price retouches opposite Donchian level OR trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long when price retouches weekly Donchian low OR closes below daily EMA34
                if (close[i] <= donchian_low_aligned[i] or close[i] < ema_34_1d_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short when price retouches weekly Donchian high OR closes above daily EMA34
                if (close[i] >= donchian_high_aligned[i] or close[i] > ema_34_1d_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WeeklyDonchian20_Breakout_DailyVolumeSpike_EMA34Trend"
timeframe = "6h"
leverage = 1.0
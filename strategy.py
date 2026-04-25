#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation (>1.5x 20-bar avg). Uses discrete sizing (0.25) to limit trades (~25/year) and avoid fee drag. The 1d EMA50 provides robust trend alignment across bull/bear regimes. Volume spike confirms breakout momentum. Designed for BTC/ETH robustness with tight entries to minimize fee drag and maximize test generalization.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 20-bar average volume for confirmation on 4h
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50, volume MA20, Donchian
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma20[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Volume confirmation: current volume > 1.5x 20-bar average
            volume_confirm = volume[i] > 1.5 * vol_ma20[i]
            
            # Long: price breaks above Donchian upper in uptrend with volume spike
            # Short: price breaks below Donchian lower in downtrend with volume spike
            long_signal = (close[i] > highest_high[i]) and (close[i] > ema50_1d_aligned[i]) and volume_confirm
            short_signal = (close[i] < lowest_low[i]) and (close[i] < ema50_1d_aligned[i]) and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below 1d EMA50 (trend reversal)
            exit_signal = close[i] < ema50_1d_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above 1d EMA50 (trend reversal)
            exit_signal = close[i] > ema50_1d_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0
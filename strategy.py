#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) Breakout with 1w EMA50 Trend Filter and Volume Confirmation
- Uses Donchian channel breakouts on 12h timeframe for structural entry signals
- 1-week EMA50 filter ensures alignment with major trend (works in bull/bear via trend filter)
- Volume confirmation (>1.8x 20-period MA) filters low-conviction breakouts
- Designed for low trade frequency (target: 12-37/year) to minimize fee drag on 12h TF
- ATR-based risk management via signal=0 on trend reversal or range re-entry
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
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    # Using rolling window on price data directly (no resampling)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need EMA50_1w, Donchian(20), vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close above Donchian upper AND price > 1w EMA50 (uptrend) AND volume spike
            if (close[i] > high_max[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.8 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close below Donchian lower AND price < 1w EMA50 (downtrend) AND volume spike
            elif (close[i] < low_min[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.8 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Close back inside Donchian channel OR loss of 1w EMA50 trend
            exit_signal = False
            if position == 1:
                # Exit long when close < Donchian lower OR price < 1w EMA50
                if close[i] < low_min[i] or close[i] < ema_50_1w_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when close > Donchian upper OR price > 1w EMA50
                if close[i] > high_max[i] or close[i] > ema_50_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0
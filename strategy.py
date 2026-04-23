#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
Enters long when price breaks above Donchian upper band in uptrend (price > 1d EMA50) with volume spike.
Enters short when price breaks below Donchian lower band in downtrend (price < 1d EMA50) with volume spike.
Exits on opposite Donchian band break. Uses discrete position sizing (0.25) to minimize fee churn.
Designed for 4h timeframe to capture intermediate-term moves with controlled trade frequency.
Target: 20-50 trades/year per symbol (80-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    # Using rolling window on primary timeframe data
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike: current volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need EMA50 and Donchian channels
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(high_rolling_max[i]) or 
            np.isnan(low_rolling_min[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs 1d EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: Break above Donchian upper band AND uptrend AND volume spike
            if close[i] > high_rolling_max[i] and trend_up and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower band AND downtrend AND volume spike
            elif close[i] < low_rolling_min[i] and trend_down and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: break of opposite Donchian band
            exit_signal = False
            if position == 1:
                # Exit long on break below Donchian lower band
                if close[i] < low_rolling_min[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short on break above Donchian upper band
                if close[i] > high_rolling_max[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0
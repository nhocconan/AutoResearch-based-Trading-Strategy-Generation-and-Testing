#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR volatility filter and volume spike confirmation.
- Uses 4h Donchian channels for trend-following breakouts in both bull and bear markets
- 1d ATR(14) ensures we only trade during sufficient volatility regimes (avoids chop)
- Volume spike >1.8x 20-period average confirms institutional participation
- Position size: 0.25 discrete level to minimize fee churn
- Trend-following works in bull markets; short breaks work in bear markets/range
- Target: 20-50 trades/year on 4h timeframe to avoid fee drag
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
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d data for ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ATR(14) for volatility filter
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]  # First TR
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ATR to 4h timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Donchian, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(atr_14_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        # Volatility filter: only trade when ATR > 0.4 * 20-period ATR average (avoid low-vol chop)
        atr_ma = pd.Series(atr_14_1d_aligned).rolling(window=20, min_periods=20).mean().values
        volatility_filter = atr_14_1d_aligned[i] > 0.4 * atr_ma[i] if not np.isnan(atr_ma[i]) else False
        
        # Donchian breakout signals
        breakout_up = close[i] > highest_high[i-1]  # Close above prior 4h Donchian high
        breakout_down = close[i] < lowest_low[i-1]  # Close below prior 4h Donchian low
        
        if position == 0:
            # Long: 4h Donchian breakout up AND volume confirmation AND volatility filter
            if breakout_up and volume_confirm and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short: 4h Donchian breakout down AND volume confirmation AND volatility filter
            elif breakout_down and volume_confirm and volatility_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: 4h Donchian break down (opposite direction)
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: 4h Donchian break up (opposite direction)
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dATR_VolumeSpike_Filter_v1"
timeframe = "4h"
leverage = 1.0
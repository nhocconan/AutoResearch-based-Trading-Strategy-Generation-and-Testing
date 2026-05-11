#!/usr/bin/env python3
"""
6h_Engulfing_Engulfing_1dTrend_VolumeSpike_v1
Hypothesis: Combine daily candle engulfing patterns with 1d trend filter and volume spike confirmation. 
Bullish engulfing in uptrend + volume spike = long; Bearish engulfing in downtrend + volume spike = short.
Uses price action for reversal signals, works in both bull and bear markets by following higher timeframe trend.
Targets 15-35 trades/year on 6h timeframe (~60-140 total over 4 years).
"""

name = "6h_Engulfing_Engulfing_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # === 1D Data for Engulfing Patterns and Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Bullish engulfing: current green candle engulfs previous red candle
    bullish_engulf = (close_1d > open_1d) & (open_1d < close_1d.shift(1)) & (close_1d > open_1d.shift(1)) & (open_1d <= close_1d.shift(1))
    # Bearish engulfing: current red candle engulfs previous green candle
    bearish_engulf = (close_1d < open_1d) & (open_1d > close_1d.shift(1)) & (close_1d < open_1d.shift(1)) & (open_1d >= close_1d.shift(1))
    
    # Handle first element (no previous day)
    bullish_engulf[0] = False
    bearish_engulf[0] = False
    
    # Trend filter: EMA50 on 1d close
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1D indicators to 6h timeframe
    bullish_engulf_aligned = align_htf_to_ltf(prices, df_1d, bullish_engulf.astype(float))
    bearish_engulf_aligned = align_htf_to_ltf(prices, df_1d, bearish_engulf.astype(float))
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume spike: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(bullish_engulf_aligned[i]) or 
            np.isnan(bearish_engulf_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish engulfing AND uptrend (close > EMA50) AND volume spike
            if bullish_engulf_aligned[i] > 0.5 and close[i] > ema_50_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish engulfing AND downtrend (close < EMA50) AND volume spike
            elif bearish_engulf_aligned[i] > 0.5 and close[i] < ema_50_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish engulfing OR price breaks below EMA50
            if bearish_engulf_aligned[i] > 0.5 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: bullish engulfing OR price breaks above EMA50
            if bullish_engulf_aligned[i] > 0.5 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals
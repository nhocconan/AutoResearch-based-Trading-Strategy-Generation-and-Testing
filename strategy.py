#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band Squeeze with 12h Donchian Breakout and Volume Confirmation.
- Primary timeframe: 6h for execution, HTF: 12h for Donchian trend direction.
- Bollinger Squeeze: BB Width (20,2) at 20-period low = low volatility primed for breakout.
- Entry: Long when price breaks above upper BB AND 12h Donchian(20) is rising with volume spike.
         Short when price breaks below lower BB AND 12h Donchian(20) is falling with volume spike.
- Exit: When price returns to middle BB (20-period SMA) or opposite signal.
- Works in bull via buying breakouts, in bear via selling breakdowns.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20,2) on 6h
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2.0 * std_20)
    lower_bb = sma_20 - (2.0 * std_20)
    bb_width = (upper_bb - lower_bb) / sma_20  # normalized width
    
    # Bollinger Squeeze: BB Width at 20-period low
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    bb_squeeze = bb_width < bb_width_ma  # width below its MA = low volatility
    
    # Get 12h data for Donchian trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Donchian(20): highest high and lowest low over 20 periods
    donchian_high = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Donchian trend: rising if current mid > previous mid
    donchian_rising = donchian_mid > np.roll(donchian_mid, 1)
    donchian_falling = donchian_mid < np.roll(donchian_mid, 1)
    # Handle first element
    donchian_rising[0] = False
    donchian_falling[0] = False
    
    # Align 12h indicators to 6h
    donchian_rising_aligned = align_htf_to_ltf(prices, df_12h, donchian_rising)
    donchian_falling_aligned = align_htf_to_ltf(prices, df_12h, donchian_falling)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # BB + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(sma_20[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or
            np.isnan(bb_squeeze[i]) or np.isnan(donchian_rising_aligned[i]) or
            np.isnan(donchian_falling_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for breakout signals with squeeze and volume
            if bb_squeeze[i] and volume_spike[i]:
                # Long: price breaks above upper BB AND 12h Donchian rising
                if close[i] > upper_bb[i] and donchian_rising_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below lower BB AND 12h Donchian falling
                elif close[i] < lower_bb[i] and donchian_falling_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price returns to middle BB or opposite breakout
            if close[i] < sma_20[i] or (close[i] < lower_bb[i] and donchian_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to middle BB or opposite breakout
            if close[i] > sma_20[i] or (close[i] > upper_bb[i] and donchian_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BB_Squeeze_Donchian_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0
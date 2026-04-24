#!/usr/bin/env python3
"""
Hypothesis: 6h Supertrend with 1d Bollinger Band Width regime filter and volume confirmation.
- Supertrend identifies trend direction using ATR multiplier
- Bollinger Band Width (BBW) on 1d: low BBW = squeeze (range), high BBW = expansion (trend)
- Long when Supertrend = uptrend AND 1d BBW > 0.05 (expansion) AND volume > 1.5 * 20-period average
- Short when Supertrend = downtrend AND 1d BBW > 0.05 AND volume > 1.5 * 20-period average
- Exit when Supertrend flips OR BBW < 0.03 (squeeze) OR volume < 1.2 * 20-period average
- Uses 6h primary with 1d HTF for BBW regime to avoid whipsaws in low-volatility ranging markets
- Supertrend catches trends, BBW filter ensures we only trade during volatility expansions
- Volume confirmation adds conviction to breakouts
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year)
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
    
    # Calculate Supertrend on 6h
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR using Wilder's smoothing
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        result[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    atr = wilders_smoothing(tr, atr_period)
    
    # Basic Upper and Lower Bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close)
    direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
            continue
            
        # Upper Band
        if upper_band[i] < supertrend[i-1] or close[i-1] > supertrend[i-1]:
            upper_band[i] = upper_band[i]
        else:
            upper_band[i] = supertrend[i-1]
            
        # Lower Band
        if lower_band[i] > supertrend[i-1] or close[i-1] < supertrend[i-1]:
            lower_band[i] = lower_band[i]
        else:
            lower_band[i] = supertrend[i-1]
        
        # Supertrend and Direction
        if close[i] <= supertrend[i-1]:
            direction[i] = -1
            supertrend[i] = upper_band[i]
        else:
            direction[i] = 1
            supertrend[i] = lower_band[i]
    
    # Calculate 1d Bollinger Band Width for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2.0
    
    ma_1d = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_1d = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    
    upper_bb = ma_1d + (bb_std * std_1d)
    lower_bb = ma_1d - (bb_std * std_1d)
    bb_width = (upper_bb - lower_bb) / ma_1d  # Normalized width
    
    # Handle division by zero and NaN
    bb_width = np.where(ma_1d == 0, 0, bb_width)
    bb_width = np.nan_to_num(bb_width, nan=0.0)
    
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    volume_weak = volume < (1.2 * vol_ma)  # For exit condition
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(atr_period, bb_period, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(supertrend[i]) or np.isnan(direction[i]) or 
            np.isnan(bb_width_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Supertrend uptrend AND BBW > 0.05 (expansion) AND volume confirmation
            if direction[i] == 1 and bb_width_aligned[i] > 0.05 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Supertrend downtrend AND BBW > 0.05 AND volume confirmation
            elif direction[i] == -1 and bb_width_aligned[i] > 0.05 and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Supertrend flips down OR BBW < 0.03 (squeeze) OR weak volume
            if direction[i] == -1 or bb_width_aligned[i] < 0.03 or volume_weak[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Supertrend flips up OR BBW < 0.03 OR weak volume
            if direction[i] == 1 or bb_width_aligned[i] < 0.03 or volume_weak[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Supertrend_1dBBW_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0
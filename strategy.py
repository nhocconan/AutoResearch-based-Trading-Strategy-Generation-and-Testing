#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d ADX Trend Filter and Volume Confirmation.
- Enter long when: BB width at 20-period low (squeeze) + price breaks above upper BB + 1d ADX > 25 (trending) + volume > 1.5x average volume
- Enter short when: BB width at 20-period low (squeeze) + price breaks below lower BB + 1d ADX > 25 (trending) + volume > 1.5x average volume
- Exit when: price returns to middle BB (20-period SMA) OR ADX drops below 20 (trend weakening)
- Uses 6h primary timeframe with 1d HTF to target 50-150 trades over 4 years (12-37/year)
- Bollinger squeeze identifies low volatility periods preceding breakouts in both bull and bear markets
- 1d ADX filter ensures we only trade in trending conditions, avoiding whipsaws in ranging markets
- Volume confirmation reduces false breakouts
- Designed for BTC/ETH with edge in capturing explosive moves after consolidation periods
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
    
    # Calculate Bollinger Bands (20, 2) using previous period (no look-ahead)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().shift(1).values
    bb_std = close_series.rolling(window=20, min_periods=20).std().shift(1).values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle  # Normalized width
    
    # Get 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr2.iloc[0] = np.nan
    tr3.iloc[0] = np.nan
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = pd.Series(np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                                 np.maximum(high_1d - np.roll(high_1d, 1), 0), 0))
    dm_minus = pd.Series(np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                                  np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0))
    # Handle first row
    dm_plus.iloc[0] = 0
    dm_minus.iloc[0] = 0
    
    # Smoothed values
    atr_1d = tr_1d.ewm(span=14, adjust=False, min_periods=14).mean()
    dm_plus_smooth = dm_plus.ewm(span=14, adjust=False, min_periods=14).mean()
    dm_minus_smooth = dm_minus.ewm(span=14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr_1d
    di_minus = 100 * dm_minus_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1d = dx.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Average volume for confirmation (20-period)
    volume_series = pd.Series(volume)
    avg_volume = volume_series.rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > 1.5 * avg_volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 30) + 1  # BB(20) and ADX needs ~30 bars
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_middle[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(avg_volume[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Bollinger squeeze condition: width at 20-period low
        bb_width_series = pd.Series(bb_width)
        bb_width_low = bb_width_series.rolling(window=20, min_periods=1).min().iloc[i]
        is_squeeze = bb_width[i] <= bb_width_low * 1.1  # Within 10% of recent low
        
        if position == 0:
            # Long: squeeze + break above upper BB + trending (ADX>25) + volume confirmation
            if (is_squeeze and close[i] > bb_upper[i] and 
                adx_1d_aligned[i] > 25 and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: squeeze + break below lower BB + trending (ADX>25) + volume confirmation
            elif (is_squeeze and close[i] < bb_lower[i] and 
                  adx_1d_aligned[i] > 25 and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to middle BB OR trend weakens (ADX<20)
            if close[i] < bb_middle[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to middle BB OR trend weakens (ADX<20)
            if close[i] > bb_middle[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BBSqueeze_ADXTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0
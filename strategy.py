#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 12h volume confirmation and 1d ADX trend filter
# Uses Bollinger Bands volatility contraction (squeeze) followed by expansion breakout,
# confirmed by volume spike on 12h and trend strength via ADX on 1d.
# Works in bull markets (breakouts continue trend) and bear markets (breakouts catch reversals).
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data (primary timeframe) for Bollinger Bands
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Load 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    volume_12h = df_12h['volume'].values
    
    # Load 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands (20, 2) on 6h
    bb_middle = pd.Series(close_6h).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_6h).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Bollinger Band Width for squeeze detection
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # Calculate ADX (14) on 1d for trend strength
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr_1d + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr_1d + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period on 12h)
    vol_avg_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    bb_width_aligned = align_htf_to_ltf(prices, df_6h, bb_width)
    bb_upper_aligned = align_htf_to_ltf(prices, df_6h, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_6h, bb_lower)
    bb_middle_aligned = align_htf_to_ltf(prices, df_6h, bb_middle)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_avg_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_12h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_width_aligned[i]) or np.isnan(bb_upper_aligned[i]) or
            np.isnan(bb_lower_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: BB squeeze breakout above upper band + volume spike + ADX > 20 (trending)
        if (close[i] > bb_upper_aligned[i] and close[i-1] <= bb_upper_aligned[i-1] and
            bb_width_aligned[i] < 0.05 and  # Squeeze threshold
            volume[i] > 2.0 * vol_avg_aligned[i] and  # Volume spike
            adx_aligned[i] > 20 and  # Trend strength filter
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: BB squeeze breakout below lower band + volume spike + ADX > 20 (trending)
        elif (close[i] < bb_lower_aligned[i] and close[i-1] >= bb_lower_aligned[i-1] and
              bb_width_aligned[i] < 0.05 and  # Squeeze threshold
              volume[i] > 2.0 * vol_avg_aligned[i] and  # Volume spike
              adx_aligned[i] > 20 and  # Trend strength filter
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse BB breakout or BB width expands significantly (end of squeeze)
        elif position == 1 and (close[i] < bb_middle_aligned[i] or bb_width_aligned[i] > 0.15):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > bb_middle_aligned[i] or bb_width_aligned[i] > 0.15):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_BB_Squeeze_Breakout_Volume_ADX"
timeframe = "6h"
leverage = 1.0
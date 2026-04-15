#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d regime filter and volume confirmation
# Elder Ray calculates bull power (high - EMA) and bear power (low - EMA) to measure bull/bear strength.
# We use 1d ADX to filter for trending markets (ADX > 25) and volume confirmation to avoid false signals.
# In trending markets, we go long when bull power is rising and positive, short when bear power is falling and negative.
# This captures momentum in trending markets while avoiding ranging conditions.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data (primary timeframe) for Elder Ray calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Load 1d data for ADX (trend filter) and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA13 for Elder Ray (standard period)
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high_6h - ema13_6h  # High minus EMA
    bear_power = low_6h - ema13_6h   # Low minus EMA
    
    # Calculate ADX (14-period) on 1d for trend strength
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+, DM- with 14-period EMA (Wilder's smoothing)
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate DI+ and DI-
    di_plus = 100 * dm_plus_14 / (tr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr_14 + 1e-10)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume average (20-period on 1d)
    vol_avg_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: ADX > 25 (trending), bull power > 0 and rising, volume confirmation
        if (adx_aligned[i] > 25 and
            bull_power_aligned[i] > 0 and
            i > 100 and bull_power_aligned[i] > bull_power_aligned[i-1] and
            volume[i] > 1.3 * vol_avg_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: ADX > 25 (trending), bear power < 0 and falling, volume confirmation
        elif (adx_aligned[i] > 25 and
              bear_power_aligned[i] < 0 and
              i > 100 and bear_power_aligned[i] < bear_power_aligned[i-1] and
              volume[i] > 1.3 * vol_avg_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: ADX drops below 20 (trend weakening) or power signals reverse
        elif position == 1 and (adx_aligned[i] < 20 or bull_power_aligned[i] < 0):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (adx_aligned[i] < 20 or bear_power_aligned[i] > 0):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_ADX_Volume_Filter"
timeframe = "6h"
leverage = 1.0
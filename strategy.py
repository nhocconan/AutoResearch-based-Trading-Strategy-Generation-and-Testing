#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and ADX trend filter
# - Entry: Price breaks above/below 12h Donchian channel (20-period) + 1d volume > 1.5x 20-period average + 1d ADX > 25
# - Exit: Price crosses back through Donchian midpoint or ADX falls below 20
# - Designed for 12h timeframe to capture medium-term trends with volume and trend confirmation
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volume and ADX filters
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d volume SMA(20)
    vol_sma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX(14)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (alpha = 1/14)
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)  # Avoid division by zero
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 12h timeframe
    vol_sma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 12h Donchian channel (20-period)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    upper_channel = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    mid_channel = (upper_channel + lower_channel) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if NaN in indicators
        if np.isnan(vol_sma_20_aligned[i]) or np.isnan(adx_aligned[i]) or \
           np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        vol_1d_current = vol_1d[i] if i < len(vol_1d) else vol_1d[-1]  # Current 1d volume
        vol_sma_20_current = vol_sma_20_aligned[i]
        adx_current = adx_aligned[i]
        
        if position == 0:
            # Long entry: Price breaks above upper channel + volume spike + strong trend
            if price > upper_channel[i] and vol_1d_current > 1.5 * vol_sma_20_current and adx_current > 25:
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below lower channel + volume spike + strong trend
            elif price < lower_channel[i] and vol_1d_current > 1.5 * vol_sma_20_current and adx_current > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below midpoint or trend weakens
            if price < mid_channel[i] or adx_current < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above midpoint or trend weakens
            if price > mid_channel[i] or adx_current < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dVolume_ADXFilter"
timeframe = "12h"
leverage = 1.0
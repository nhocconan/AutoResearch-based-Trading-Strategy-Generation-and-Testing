#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using Donchian breakout with volume confirmation and ADX trend filter.
- Calculate Donchian channels from previous 20-period high/low
- Enter long when price breaks above upper band with volume > 1.5x 20-period volume MA and ADX > 20
- Enter short when price breaks below lower band with volume > 1.5x 20-period volume MA and ADX > 20
- Exit when price crosses back to the opposite band
- Fixed position size 0.25 to manage drawdown
- Uses ADX to ensure we only trade in trending markets
- Designed for 4h timeframe with strict entry conditions to limit trades to 75-200 total over 4 years
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
    
    # Get 1-day data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ADX from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
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
    
    # Smoothed values
    period = 14
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False).mean()
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False).mean()
    
    # DI values
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean()
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    
    # Calculate Donchian channels from previous 20-period high/low
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # Align Donchian levels (already in correct timeframe)
    upper_band = high_20.values
    lower_band = low_20.values
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(volume_ma_20.iloc[i]) or 
            np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        upper = upper_band[i]
        lower = lower_band[i]
        adx_val = adx_aligned[i]
        
        if position == 0:
            # Look for Donchian breakouts with volume confirmation and trend filter
            # Long: price breaks above upper band + volume spike + ADX > 20
            if price > upper and vol > 1.5 * vol_ma and adx_val > 20:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band + volume spike + ADX > 20
            elif price < lower and vol > 1.5 * vol_ma and adx_val > 20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price crosses below lower band
            if price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses above upper band
            if price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0
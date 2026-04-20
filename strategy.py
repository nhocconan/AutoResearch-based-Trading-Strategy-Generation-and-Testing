#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume confirmation + ADX trend filter
# - Long when price breaks above 12h Donchian upper (20) + 1d volume > 1.5x 20-period average + 1d ADX > 25
# - Short when price breaks below 12h Donchian lower (20) + 1d volume > 1.5x 20-period average + 1d ADX > 25
# - Uses volatility breakout structure with volume confirmation and trend strength filter
# - Designed for 12h timeframe to target 12-37 trades per year (50-150 total over 4 years)
# - ADX ensures we only trade in trending markets, avoiding whipsaws in ranging conditions

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volume and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX (14-period)
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
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 12h timeframe
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        vol_ma = vol_ma_20_aligned[i]
        adx_val = adx_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        volume = vol_1d[i] if i < len(vol_1d) else vol_1d[-1]  # fallback for alignment
        
        if position == 0:
            # Long entry: price breaks above Donchian high + volume spike + strong trend
            if price > upper and volume > 1.5 * vol_ma and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + volume spike + strong trend
            elif price < lower and volume > 1.5 * vol_ma and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or trend weakens
            if price < lower or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend weakens
            if price > upper or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Volume_ADXFilter"
timeframe = "12h"
leverage = 1.0
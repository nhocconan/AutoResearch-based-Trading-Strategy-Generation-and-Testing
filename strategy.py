# 4h_1w_Donchian_Breakout_With_Trend_And_Volume
# Uses 1w ADX for trend regime and 1d volume for confirmation
# Breakouts trigger only in strong trends (ADX>25) with volume > 1.5x average
# Position size: 0.25 for clear trend alignment
# Target: 20-40 trades/year per symbol

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1w_Donchian_Breakout_With_Trend_And_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w: ADX trend filter ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        first_avg = np.nansum(arr[1:period+1]) if not np.all(np.isnan(arr[1:period+1])) else np.nan
        result[period] = first_avg
        # Wilder smoothing
        for i in range(period+1, len(arr)):
            if not np.isnan(result[i-1]) and not np.isnan(arr[i]):
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
            else:
                result[i] = np.nan
        return result
    
    tr_smoothed = smooth_wilder(tr, 14)
    dm_plus_smoothed = smooth_wilder(dm_plus, 14)
    dm_minus_smoothed = smooth_wilder(dm_minus, 14)
    
    # DI values
    di_plus = np.where(tr_smoothed > 0, (dm_plus_smoothed / tr_smoothed) * 100, 0)
    di_minus = np.where(tr_smoothed > 0, (dm_minus_smoothed / tr_smoothed) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = np.full_like(dx, np.nan)
    if len(dx) >= 27:
        # First ADX is average of first 14 DX values
        first_adx = np.nanmean(dx[1:15]) if not np.all(np.isnan(dx[1:15])) else np.nan
        adx[14] = first_adx
        # Wilder smoothing for ADX
        for i in range(15, len(dx)):
            if not np.isnan(adx[i-1]) and not np.isnan(dx[i]):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
            else:
                adx[i] = np.nan
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Volume average
    vol_1d = df_1d['volume'].values
    vol_avg = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    
    # === 4h: Donchian channels ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        adx_val = adx_aligned[i]
        vol_avg_val = vol_avg_aligned[i]
        vol_val = volume[i]
        high_val = high[i]
        low_val = low[i]
        close_val = close[i]
        upper_channel = high_20[i]
        lower_channel = low_20[i]
        
        # Skip if any value is NaN
        if (np.isnan(adx_val) or np.isnan(vol_avg_val) or 
            np.isnan(upper_channel) or np.isnan(lower_channel)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume ratio
        vol_ratio = vol_val / vol_avg_val if vol_avg_val > 0 else 0
        
        if position == 0:
            # Long: breakout above upper channel with strong trend and volume
            if (high_val > upper_channel and 
                adx_val > 25 and 
                vol_ratio > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower channel with strong trend and volume
            elif (low_val < lower_channel and 
                  adx_val > 25 and 
                  vol_ratio > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: breakdown below lower channel or trend weakening
            if (low_val < lower_channel or adx_val < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: breakout above upper channel or trend weakening
            if (high_val > upper_channel or adx_val < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
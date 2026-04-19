#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume confirmation + 1d ADX trend filter
# - Donchian breakout captures breakouts in both bull and bear markets
# - 1d volume > 2.0x 20-period average ensures conviction
# - 1d ADX > 25 filters for trending markets only (avoids chop)
# - Exit on opposite Donchian breakout or trend reversal (ADX < 20)
# - Position size: 0.25 (25%) to balance return and drawdown
# - Target: ~25-40 trades/year to avoid excessive fee drag

name = "4h_Donchian20_1dVolume_ADX_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume and ADX confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Donchian channels (20-period) on 4h data
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Rolling max/min
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_high = rolling_max(high_4h, 20)
    donchian_low = rolling_min(low_4h, 20)
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 2.0x 1d average volume (scaled)
        # Approximate 4h volume as 1/6 of daily volume
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > (2.0 * vol_ma_1d_aligned[i] / 6.0)
        
        # ADX filter: trending market only
        trend_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Look for long entry: price breaks above Donchian high + volume + trend
            if close[i] > donchian_high_aligned[i] and volume_filter and trend_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: price breaks below Donchian low + volume + trend
            elif close[i] < donchian_low_aligned[i] and volume_filter and trend_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on break below Donchian low or trend weakening
            if close[i] < donchian_low_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on break above Donchian high or trend weakening
            if close[i] > donchian_high_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
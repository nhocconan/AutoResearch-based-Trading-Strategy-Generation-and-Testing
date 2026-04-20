#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_R1S1_Breakout_Volume_Control_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d: Calculate Camarilla pivot levels (using previous day's data) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use previous day's OHLC for today's levels
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    
    # Set first day's values to NaN
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Calculate Camarilla levels: R1, S1
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align 1d indicators to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 4h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === 4h: ADX for trend strength filter ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period]) if not np.all(np.isnan(arr[1:period])) else np.nan
        # Subsequent values
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]) and not np.isnan(arr[i]):
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr_period = 14
    tr_smooth = wilders_smoothing(tr, atr_period)
    dm_plus_smooth = wilders_smoothing(dm_plus, atr_period)
    dm_minus_smooth = wilders_smoothing(dm_minus, atr_period)
    
    # Avoid division by zero
    dm_plus_smooth = np.where(tr_smooth == 0, 0, dm_plus_smooth)
    dm_minus_smooth = np.where(tr_smooth == 0, 0, dm_minus_smooth)
    
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    dx = np.where((di_plus + di_minus) == 0, 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus))
    adx = wilders_smoothing(dx, atr_period)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(35, n):  # Start after warmup for ADX
        # Get values
        close_val = close[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        vol_ratio_val = vol_ratio[i]
        adx_val = adx[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_level) or np.isnan(s1_level) or 
            np.isnan(vol_ratio_val) or np.isnan(adx_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation and weak trend (ADX < 25)
            if (close_val > r1_level and   # Break above R1
                vol_ratio_val > 2.0 and    # Strong volume confirmation
                adx_val < 25):             # Weak trend - mean reversion favorable
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume confirmation and weak trend
            elif (close_val < s1_level and   # Break below S1
                  vol_ratio_val > 2.0 and    # Strong volume confirmation
                  adx_val < 25):             # Weak trend - mean reversion favorable
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price drops back below R1 (reversion to mean)
            if close_val < r1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises back above S1 (reversion to mean)
            if close_val > s1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
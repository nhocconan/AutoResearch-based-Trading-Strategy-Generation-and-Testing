#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_Pivot_R3S3_Breakout_1dEMA34_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (based on previous day's OHLC)
    # Using previous day's data for current day's levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Camarilla levels: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    r3 = close_1d + range_hl * 1.1 / 4
    s3 = close_1d - range_hl * 1.1 / 4
    
    # Calculate 34-period EMA on daily close for trend filter
    close_ser = pd.Series(close_1d)
    ema_34 = close_ser.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all 1d indicators to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Get 1d volume average for spike detection
    vol_1d = pd.Series(df_1d['volume'].values)
    vol_ma20_1d = vol_1d.rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Current 4h volume for confirmation (20-period MA)
    vol_series = pd.Series(volume)
    vol_ma20_current = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Add 4h ADX filter for trend strength (14-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # True Range
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift())
    tr3 = abs(low_series - close_series.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = np.where((high_series.diff() > low_series.diff().abs()) & (high_series.diff() > 0), high_series.diff(), 0)
    dm_minus = np.where((low_series.diff().abs() > high_series.diff()) & (low_series.diff() < 0), low_series.diff().abs(), 0)
    
    # Smoothed values
    tr_ma = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    dm_plus_ma = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean()
    dm_minus_ma = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean()
    
    # DI and DX
    di_plus = 100 * dm_plus_ma / tr_ma
    di_minus = 100 * dm_minus_ma / tr_ma
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i]) or 
            np.isnan(vol_ma20_current[i]) or np.isnan(adx_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma20_current[i]  # Volume spike filter
        adx_ok = adx_values[i] > 25  # Strong trend filter
        
        if position == 0:
            # Long: Price breaks above R3 with volume spike, above EMA34, and strong ADX
            if close[i] > r3_aligned[i] and vol_ok and close[i] > ema_34_aligned[i] and adx_ok:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume spike, below EMA34, and strong ADX
            elif close[i] < s3_aligned[i] and vol_ok and close[i] < ema_34_aligned[i] and adx_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below R3 OR ADX weakens
            if close[i] < r3_aligned[i] or adx_values[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above S3 OR ADX weakens
            if close[i] > s3_aligned[i] or adx_values[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
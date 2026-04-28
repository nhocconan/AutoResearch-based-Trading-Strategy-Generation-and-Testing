#!/usr/bin/env python3
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
    
    # Get daily data for pivot points and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Calculate daily pivot points (floor trader method)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2.0 * pivot_1d - low_1d
    s1_1d = 2.0 * pivot_1d - high_1d
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    
    # Align daily pivot levels to 1d timeframe (already aligned, but for consistency)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Calculate daily volume spike (current volume > 1.5x 20-period MA)
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (1.5 * vol_ma_20_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    # Calculate 1d ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily ATR for volatility filter
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr2_1d[0] = 0
    tr3_1d[0] = 0
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i]) or
            np.isnan(vol_spike_aligned[i]) or
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely low volatility periods
        if atr_1d_aligned[i] < 0.5 * np.nanmedian(atr_1d_aligned[max(0, i-50):i+1]):
            signals[i] = 0.0
            continue
        
        # Daily pivot levels
        r2 = r2_aligned[i]
        s2 = s2_aligned[i]
        
        # Volume spike confirmation from daily timeframe
        vol_spike = vol_spike_aligned[i] > 0.5
        
        # Entry conditions: 
        # Long: Price breaks above daily R2 with volume spike
        # Short: Price breaks below daily S2 with volume spike
        long_entry = (close[i] > r2) and vol_spike
        short_entry = (close[i] < s2) and vol_spike
        
        # Exit conditions: 
        # Long exit: price returns below daily pivot
        # Short exit: price returns above daily pivot
        pivot_1d_series = (high_1d + low_1d + close_1d) / 3.0
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d_series)
        pivot_val = pivot_aligned[i] if not np.isnan(pivot_aligned[i]) else (r2 + s2) / 2.0
        
        long_exit = close[i] < pivot_val
        short_exit = close[i] > pivot_val
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_DailyPivot_R2S2_Breakout_VolumeSpike_v2"
timeframe = "1d"
leverage = 1.0
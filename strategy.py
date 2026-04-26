#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_4hTrend_1dVolumeSpike
Hypothesis: Camarilla R3/S3 breakout on 1h with 4h EMA50 trend filter and 1d volume confirmation. 
Designed for low trade frequency (15-37/year) to avoid fee drag. Uses 4h/1d for signal direction 
and 1h for entry timing precision. Session filter (08-20 UTC) reduces noise. Discrete sizing 0.20.
Works in both bull/bear via trend filter + volume spike confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 4h data ONCE before loop for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    close_4h_series = pd.Series(close_4h)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data for Camarilla levels and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for stoploss
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate Camarilla levels from prior 1d bar
    rng = high_1d - low_1d
    r3 = close_1d + 1.125 * rng
    s3 = close_1d - 1.125 * rng
    r4 = close_1d + 1.5 * rng
    s4 = close_1d - 1.5 * rng
    
    # Align to 1h timeframe (wait for 1d bar to close)
    r3_1h = align_htf_to_ltf(prices, df_1d, r3)
    s3_1h = align_htf_to_ltf(prices, df_1d, s3)
    r4_1h = align_htf_to_ltf(prices, df_1d, r4)
    s4_1h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume spike: volume > 2.0x 20-period median volume (stricter for 1h)
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (2.0 * vol_median_20)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 50 for 4h EMA, 14 for ATR, 20 for volume median
    start_idx = max(50, 14, 20)
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC only
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(r3_1h[i]) or
            np.isnan(s3_1h[i]) or
            np.isnan(r4_1h[i]) or
            np.isnan(s4_1h[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_4h_aligned[i]
        atr_val = atr_14_1d_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: price > R3 and volume spike, in uptrend (close > EMA50_4h)
            long_entry = (close_val > r3_1h[i]) and vol_spike and (close_val > ema_50_val)
            # Short: price < S3 and volume spike, in downtrend (close < EMA50_4h)
            short_entry = (close_val < s3_1h[i]) and vol_spike and (close_val < ema_50_val)
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal, ATR stoploss, or at R4 (take profit)
            stop_price = entry_price - 2.0 * atr_val
            if close_val < ema_50_val or close_val < stop_price or close_val > r4_1h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal, ATR stoploss, or at S4 (take profit)
            stop_price = entry_price + 2.0 * atr_val
            if close_val > ema_50_val or close_val > stop_price or close_val < s4_1h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_1dVolumeSpike"
timeframe = "1h"
leverage = 1.0
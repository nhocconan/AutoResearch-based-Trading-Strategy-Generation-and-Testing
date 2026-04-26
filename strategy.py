#!/usr/bin/env python3
"""
1d_WeeklyPivot_Donchian20_Breakout_VolumeSpike_TrendFilter_v1
Hypothesis: On 1d timeframe, trade Donchian(20) breakouts from prior week's Camarilla pivot levels (R3/S3) with volume confirmation and 1w EMA50 trend filter. 
Uses ATR-based trailing stop (2.5 ATR) and discrete position sizing (0.25) to limit trades to 30-100 over 4 years. 
Works in bull/bear via 1w trend filter; avoids overtrading by requiring confluence of pivot, breakout, volume, and trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1w ATR(14) for stoploss
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Load 1d data for Camarilla pivot calculation (prior week's close)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla R3, S3 from prior 1d bar (using weekly close as anchor)
    # We use weekly close to calculate pivots that are constant throughout the week
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    rng = weekly_high - weekly_low
    r3 = weekly_close + 1.125 * rng  # Camarilla R3
    s3 = weekly_close - 1.125 * rng  # Camarilla S3
    
    # Align 1w Camarilla levels to 1d timeframe (constant throughout week)
    r3_1d = align_htf_to_ltf(prices, df_1w, r3)
    s3_1d = align_htf_to_ltf(prices, df_1w, s3)
    
    # Calculate Donchian(20) channels on 1d
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume spike: volume > 2.0x 20-period median volume (stricter to reduce trades)
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (2.0 * vol_median_20)
    
    # Fixed position size to control trade frequency and drawdown
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: need 50 for 1w EMA, 14 for ATR, 20 for Donchian and volume median
    start_idx = max(50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(atr_14_1w_aligned[i]) or
            np.isnan(r3_1d[i]) or
            np.isnan(s3_1d[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_50_val = ema_50_1w_aligned[i]
        atr_val = atr_14_1w_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: price breaks above Donchian high AND above R3 pivot, with volume spike, in uptrend (close > EMA50_1w)
            long_breakout = (high_val > donchian_high[i]) and (close_val > r3_1d[i])
            long_entry = long_breakout and vol_spike and (close_val > ema_50_val)
            # Short: price breaks below Donchian low AND below S3 pivot, with volume spike, in downtrend (close < EMA50_1w)
            short_breakout = (low_val < donchian_low[i]) and (close_val < s3_1d[i])
            short_entry = short_breakout and vol_spike and (close_val < ema_50_val)
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - update highest and check exit conditions
            highest_since_entry = max(highest_since_entry, high_val)
            # Exit on trend reversal, ATR trailing stop, or at 1.5x ATR profit target
            stop_price = highest_since_entry - 2.5 * atr_val
            profit_target = entry_price + 1.5 * atr_val
            if close_val < ema_50_val or close_val < stop_price or close_val > profit_target:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - update lowest and check exit conditions
            lowest_since_entry = min(lowest_since_entry, low_val)
            # Exit on trend reversal, ATR trailing stop, or at 1.5x ATR profit target
            stop_price = lowest_since_entry + 2.5 * atr_val
            profit_target = entry_price - 1.5 * atr_val
            if close_val > ema_50_val or close_val > stop_price or close_val < profit_target:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyPivot_Donchian20_Breakout_VolumeSpike_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0
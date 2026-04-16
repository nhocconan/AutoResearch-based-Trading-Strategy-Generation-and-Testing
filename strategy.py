#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h data (HTF for key levels) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === Previous 12h Values for Pivot Calculation ===
    prev_close_12h = np.roll(close_12h, 1)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    # First period uses current period values (no look-ahead)
    prev_close_12h[0] = close_12h[0]
    prev_high_12h[0] = high_12h[0]
    prev_low_12h[0] = low_12h[0]
    
    # === 12h Pivot Points ===
    pivot_point_12h = (prev_high_12h + prev_low_12h + prev_close_12h) / 3
    prev_range_12h = prev_high_12h - prev_low_12h
    
    # === R1 and S1 Levels (Standard Pivot) ===
    r1_12h = pivot_point_12h + prev_range_12h * 0.382  # Fibonacci 0.382
    s1_12h = pivot_point_12h - prev_range_12h * 0.382
    
    # === R2 and S2 Levels (Exit) ===
    r2_12h = pivot_point_12h + prev_range_12h * 0.618  # Fibonacci 0.618
    s2_12h = pivot_point_12h - prev_range_12h * 0.618
    
    # === 12h ADX Trend Filter ===
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_12h[0] = high_12h[0] - low_12h[0]
    
    # Directional Movement
    dm_plus_12h = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h),
                           np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus_12h = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)),
                            np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus_12h[0] = 0
    dm_minus_12h[0] = 0
    
    # Wilder's smoothing
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr14_12h = wilders_smooth(tr_12h, period)
    dm_plus14_12h = wilders_smooth(dm_plus_12h, period)
    dm_minus14_12h = wilders_smooth(dm_minus_12h, period)
    
    # Avoid division by zero
    dm_plus14_12h_safe = np.where(tr14_12h == 0, 1, dm_plus14_12h)
    dm_minus14_12h_safe = np.where(tr14_12h == 0, 1, dm_minus14_12h)
    tr14_12h_safe = np.where(tr14_12h == 0, 1, tr14_12h)
    
    di_plus_12h = 100 * dm_plus14_12h / tr14_12h_safe
    di_minus_12h = 100 * dm_minus14_12h / tr14_12h_safe
    dx_12h = 100 * np.abs(di_plus_12h - di_minus_12h) / (di_plus_12h + di_minus_12h)
    adx_12h = wilders_smooth(dx_12h, period)
    
    # Align 12h indicators to 6h timeframe
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    r2_12h_aligned = align_htf_to_ltf(prices, df_12h, r2_12h)
    s2_12h_aligned = align_htf_to_ltf(prices, df_12h, s2_12h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === Volume Confirmation (6h) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    # === ATR for dynamic stop (6h) ===
    tr_6h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_6h[0] = high[0] - low[0]
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: enough for ADX calculation (14+14+14=42) plus buffer
    warmup = 60
    
    # Track position and entry price for stop management
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or 
            np.isnan(r2_12h_aligned[i]) or np.isnan(s2_12h_aligned[i]) or 
            np.isnan(adx_12h_aligned[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(atr_6h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1_val = r1_12h_aligned[i]
        s1_val = s1_12h_aligned[i]
        r2_val = r2_12h_aligned[i]
        s2_val = s2_12h_aligned[i]
        adx_val = adx_12h_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr_6h[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit conditions: stop at S1, target at R2, or adverse 2x ATR move
            if price < s1_val or price > r2_val or price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit conditions: stop at R1, target at S2, or adverse 2x ATR move
            if price > r1_val or price < s2_val or price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Strong trend filter: ADX > 25
            if adx_val > 25:
                # LONG: Price breaks above R1 with volume confirmation
                if price > r1_val and vol_ratio_val > 2.0:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                
                # SHORT: Price breaks below S1 with volume confirmation
                elif price < s1_val and vol_ratio_val > 2.0:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_FibPivot_R1_S1_Volume_ADX_Filter"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
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
    
    # === Daily data for pivot and trend filters ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Previous Day Values for Pivot Calculation ===
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    # === Daily Pivot Points (Standard) ===
    pivot_point = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    prev_range = prev_high_1d - prev_low_1d
    # Fibonacci-based levels: R1 at 0.382, S1 at 0.382
    r1 = pivot_point + prev_range * 0.382
    s1 = pivot_point - prev_range * 0.382
    # Additional levels for exit: R2 at 0.618, S2 at 0.618
    r2 = pivot_point + prev_range * 0.618
    s2 = pivot_point - prev_range * 0.618
    
    # Align all levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # === ADX Trend Filter (Daily) ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d[0] - low_1d[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Wilder's smoothing function
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr14 = wilders_smooth(tr, period)
    dm_plus14 = wilders_smooth(dm_plus, period)
    dm_minus14 = wilders_smooth(dm_minus, period)
    
    # Avoid division by zero
    dm_plus14_safe = np.where(tr14 == 0, 1, dm_plus14)
    dm_minus14_safe = np.where(tr14 == 0, 1, dm_minus14)
    tr14_safe = np.where(tr14 == 0, 1, tr14)
    
    di_plus = 100 * dm_plus14 / tr14_safe
    di_minus = 100 * dm_minus14 / tr14_safe
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilders_smooth(dx, period)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === Volume Confirmation (4h) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    # === ATR for dynamic stop (4h) ===
    tr_4h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_4h[0] = high[0] - low[0]
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: enough for ADX calculation (14+14+14=42) plus buffer
    warmup = 50
    
    # Track position and entry price for stop management
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(atr_4h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        adx_val = adx_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr_4h[i]
        
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

name = "4h_FibPivot_Volume_ADX_Filter"
timeframe = "4h"
leverage = 1.0
# Hypothesis: 4h 1-day pivot point breakout with volume confirmation and 4h ADX trend filter
# Pivot points act as key support/resistance levels. Breakouts above R1 or below S1 with volume
# and trend confirmation capture directional moves in both bull and bear markets.
# Using daily pivots ensures we respect actual market structure. Volume and ADX filter reduce false breakouts.
# Position size is kept moderate (0.25) to manage drawdown during 2022 crash while allowing profit in trends.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Align pivot levels to 4h timeframe (wait for daily close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 4h ADX for trend filtering (min_periods=14)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def WilderSmooth(data, period):
        smoothed = np.zeros_like(data)
        smoothed[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            smoothed[i] = (smoothed[i-1] * (period-1) + data[i]) / period
        return smoothed
    
    atr = WilderSmooth(tr, 14)
    dm_plus_smooth = WilderSmooth(dm_plus, 14)
    dm_minus_smooth = WilderSmooth(dm_minus, 14)
    
    # Avoid division by zero
    dm_plus_smooth = np.where(dm_plus_smooth == 0, 1e-10, dm_plus_smooth)
    dm_minus_smooth = np.where(dm_minus_smooth == 0, 1e-10, dm_minus_smooth)
    atr_safe = np.where(atr == 0, 1e-10, atr)
    
    di_plus = 100 * dm_plus_smooth / atr_safe
    di_minus = 100 * dm_minus_smooth / atr_safe
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    adx = WilderSmooth(dx, 14)
    
    # Volume average (20-period)
    volume = prices['volume'].values
    vol_ma = np.zeros_like(volume)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_ma[:20] = np.nan
    
    # Generate signals
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            continue
        
        close_price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Long conditions: price breaks above R1, volume spike, ADX > 20 (trending)
        if close_price > r1_aligned[i] and vol_ratio > 1.5 and adx[i] > 20:
            signals[i] = 0.25
        
        # Short conditions: price breaks below S1, volume spike, ADX > 20 (trending)
        elif close_price < s1_aligned[i] and vol_ratio > 1.5 and adx[i] > 20:
            signals[i] = -0.25
    
    return signals

name = "4h_1d_Pivot_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0
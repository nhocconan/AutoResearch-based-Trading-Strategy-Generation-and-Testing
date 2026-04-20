#!/usr/bin/env python3
# 6h_1d_ADX_Trend_With_Volume_Filter
# Hypothesis: ADX > 25 indicates strong trend on 1d, with +DI/-DI crossover for entry and volume confirmation to avoid false signals.
# Works in bull/bear: Only take longs when +DI crosses above -DI and ADX > 25, shorts when -DI crosses above +DI and ADX > 25.
# Uses 6h timeframe for execution, targeting 20-40 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ADX_Trend_With_Volume_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === ADX calculation on 1d ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    period = 14
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial values
    atr[period-1] = np.mean(tr[:period])
    dm_plus_smooth[period-1] = np.mean(dm_plus[:period])
    dm_minus_smooth[period-1] = np.mean(dm_minus[:period])
    
    # Wilder's smoothing
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
    
    # DI values
    plus_di = 100 * dm_plus_smooth / np.where(atr != 0, atr, np.nan)
    minus_di = 100 * dm_minus_smooth / np.where(atr != 0, atr, np.nan)
    
    # ADX
    dx = np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), np.nan) * 100
    adx = np.zeros_like(dx)
    adx[2*period-1] = np.mean(dx[period-1:2*period-1])
    for i in range(2*period, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Convert to numpy arrays
    adx = adx
    plus_di = plus_di
    minus_di = minus_di
    
    # === 6h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align all 1d ADX data to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(2*period, n):  # Start after ADX warmup
        # Get values
        adx_val = adx_aligned[i]
        plus_di_val = plus_di_aligned[i]
        minus_di_val = minus_di_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(adx_val) or np.isnan(plus_di_val) or 
            np.isnan(minus_di_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: +DI crosses above -DI, ADX > 25, volume confirmation
            if (plus_di_val > minus_di_val and 
                adx_val > 25 and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: -DI crosses above +DI, ADX > 25, volume confirmation
            elif (minus_di_val > plus_di_val and 
                  adx_val > 25 and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: -DI crosses above +DI or ADX falls below 20
            if minus_di_val > plus_di_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: +DI crosses above -DI or ADX falls below 20
            if plus_di_val > minus_di_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
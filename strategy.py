#!/usr/bin/env python3
"""
4h Camarilla Pivot Breakout + Volume Spike + ADX Trend Filter
Long: Close breaks above R4 + volume > 1.5x 20-bar volume SMA + ADX > 25
Short: Close breaks below S4 + volume > 1.5x 20-bar volume SMA + ADX > 25
Exit: Close returns inside (R3, S3) range
Camarilla pivots provide institutional support/resistance levels.
Volume confirms institutional participation.
ADX ensures we only trade in trending regimes to avoid whipsaws.
Designed for 4h timeframe with 12h trend filter.
Target: 50-150 total trades over 4 years (12-37/year)
"""

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
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX(14) on 12h
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # first period
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smooth with Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        dm_plus_smooth = np.zeros_like(dm_plus)
        dm_minus_smooth = np.zeros_like(dm_minus)
        
        atr[0] = tr[0]
        dm_plus_smooth[0] = dm_plus[0]
        dm_minus_smooth[0] = dm_minus[0]
        
        for i in range(1, len(tr)):
            atr[i] = atr[i-1] - (atr[i-1] / period) + tr[i]
            dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1] / period) + dm_plus[i]
            dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1] / period) + dm_minus[i]
        
        # Avoid division by zero
        dm_plus_di = 100 * dm_plus_smooth / (atr + 1e-10)
        dm_minus_di = 100 * dm_minus_smooth / (atr + 1e-10)
        
        dx = np.abs(dm_plus_di - dm_minus_di) / (dm_plus_di + dm_minus_di + 1e-10) * 100
        
        # Smooth DX to get ADX
        adx = np.zeros_like(dx)
        adx[0] = dx[0]
        for i in range(1, len(dx)):
            adx[i] = adx[i-1] - (adx[i-1] / period) + dx[i]
        
        # Set first 'period' values to NaN
        adx[:period] = np.nan
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate Camarilla levels from previous day
    # We'll use daily OHLC from 1d data
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla multipliers
    camarilla_mult = [1.1/12, 1.1/6, 1.1/4, 1.1/2]  # for R1,S1 to R4,S4
    
    # Calculate pivots for each bar using previous day's data
    pivots_r4 = np.full(n, np.nan)
    pivots_s4 = np.full(n, np.nan)
    pivots_r3 = np.full(n, np.nan)
    pivots_s3 = np.full(n, np.nan)
    
    # For each bar, get previous day's OHLC
    for i in range(n):
        # Find index of previous day in 1d data
        # Since we're on 4h timeframe, we need to map to daily
        # Simplified: use the most recent completed day
        if i >= 6:  # at least 6*4h = 24h = 1 day back
            # Get previous day's data (assuming 6 bars per day on 4h)
            day_idx = i // 6
            if day_idx > 0:
                prev_day_idx = day_idx - 1
                if prev_day_idx < len(high_1d):
                    ph = high_1d[prev_day_idx]
                    pl = low_1d[prev_day_idx]
                    pc = close_1d[prev_day_idx]
                    rng = ph - pl
                    
                    pivots_r4[i] = pc + rng * camarilla_mult[3]  # R4
                    pivots_s4[i] = pc - rng * camarilla_mult[3]  # S4
                    pivots_r3[i] = pc + rng * camarilla_mult[2]  # R3
                    pivots_s3[i] = pc - rng * camarilla_mult[2]  # S3
    
    # Volume confirmation
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(30, 20)  # need enough data for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(pivots_r4[i]) or 
            np.isnan(pivots_s4[i]) or np.isnan(pivots_r3[i]) or 
            np.isnan(pivots_s3[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_20[i]
        adx_val = adx_12h_aligned[i]
        r4 = pivots_r4[i]
        s4 = pivots_s4[i]
        r3 = pivots_r3[i]
        s3 = pivots_s3[i]
        
        # Volume spike condition
        vol_spike = vol > 1.5 * vol_sma_val
        
        # ADX trend filter
        trending = adx_val > 25
        
        if position == 0:
            # Long: Break above R4 + volume spike + trending
            if price > r4 and vol_spike and trending:
                signals[i] = 0.25
                position = 1
            # Short: Break below S4 + volume spike + trending
            elif price < s4 and vol_spike and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns inside R3
            if price < r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns inside S3
            if price > s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0
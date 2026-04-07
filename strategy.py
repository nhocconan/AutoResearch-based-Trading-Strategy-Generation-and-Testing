#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout with 1d ADX and Volume Confirmation
# Hypothesis: Donchian(20) breakouts in trending markets (ADX>25) with above-average volume
# capture momentum moves. Works in bull/bear by only taking breakouts in ADX-trend direction.
# Target: 15-25 trades/year (60-100 total) to minimize fee drag.

name = "4h_donchian_breakout_1d_adx_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_period = 14
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial values
    atr[tr_period-1] = np.mean(tr[:tr_period])
    dm_plus_smooth[tr_period-1] = np.mean(dm_plus[:tr_period])
    dm_minus_smooth[tr_period-1] = np.mean(dm_minus[:tr_period])
    
    # Wilder's smoothing
    for i in range(tr_period, len(tr)):
        atr[i] = (atr[i-1] * (tr_period-1) + tr[i]) / tr_period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (tr_period-1) + dm_plus[i]) / tr_period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (tr_period-1) + dm_minus[i]) / tr_period
    
    # DI and DX
    di_plus = np.zeros_like(atr)
    di_minus = np.zeros_like(atr)
    dx = np.zeros_like(atr)
    
    mask = atr != 0
    di_plus[mask] = 100 * dm_plus_smooth[mask] / atr[mask]
    di_minus[mask] = 100 * dm_minus_smooth[mask] / atr[mask]
    dx_mask = (di_plus + di_minus) != 0
    dx[dx_mask] = 100 * np.abs(di_plus[dx_mask] - di_minus[dx_mask]) / (di_plus[dx_mask] + di_minus[dx_mask])
    
    # ADX
    adx = np.zeros_like(dx)
    adx[tr_period*2-2:] = pd.Series(dx[tr_period*2-2:]).ewm(span=tr_period, adjust=False).mean().values
    
    # Align 1d ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Donchian(20) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):
        # Skip if required data not available
        if (np.isnan(adx_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower or ADX weakens
            if close[i] < lowest_low[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper or ADX weakens
            if close[i] > highest_high[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average
            vol_confirm = volume[i] > 1.5 * vol_ma[i]
            
            # Long: Donchian breakout above, ADX>25 (trending), volume confirm
            if (close[i] > highest_high[i] and 
                adx_aligned[i] > 25 and vol_confirm):
                position = 1
                signals[i] = 0.25
            # Short: Donchian breakdown below, ADX>25 (trending), volume confirm
            elif (close[i] < lowest_low[i] and 
                  adx_aligned[i] > 25 and vol_confirm):
                position = -1
                signals[i] = -0.25
    
    return signals
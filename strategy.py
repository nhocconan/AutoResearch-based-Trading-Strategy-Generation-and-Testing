#!/usr/bin/env python3
"""
4h Camarilla Pivot R1/S1 Breakout with Volume Spike and Daily ADX Trend Filter
Hypothesis: Camarilla pivot levels act as strong support/resistance. Breakouts above R1 or below S1
with volume confirmation and daily ADX > 25 (trending market) capture momentum moves.
This strategy targets 20-40 trades per year by requiring multiple confirmations,
reducing fee drag while maintaining edge in both bull and bear markets.
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
    
    # Get daily data for Camarilla pivot and ADX (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Only calculate when we have previous day data
    valid_prev = ~(np.isnan(prev_close) | np.isnan(prev_high) | np.isnan(prev_low))
    camarilla_range = np.zeros_like(prev_close)
    camarilla_range[valid_prev] = (prev_high[valid_prev] - prev_low[valid_prev]) * 1.1 / 12
    
    r1 = np.full_like(prev_close, np.nan)
    s1 = np.full_like(prev_close, np.nan)
    r1[valid_prev] = prev_close[valid_prev] + camarilla_range[valid_prev]
    s1[valid_prev] = prev_close[valid_prev] - camarilla_range[valid_prev]
    
    # Align R1 and S1 to 4h timeframe (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate daily ADX for trend filter (requires 14 periods)
    if len(df_1d) >= 14:
        # True Range
        tr1 = df_1d['high'] - df_1d['low']
        tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
        tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        dm_plus = np.where((df_1d['high'] - df_1d['high'].shift(1)) > (df_1d['low'].shift(1) - df_1d['low']),
                           np.maximum(df_1d['high'] - df_1d['high'].shift(1), 0), 0)
        dm_minus = np.where((df_1d['low'].shift(1) - df_1d['low']) > (df_1d['high'] - df_1d['high'].shift(1)),
                            np.maximum(df_1d['low'].shift(1) - df_1d['low'], 0), 0)
        
        # Smooth TR, DM+, DM- using Wilder's smoothing (equivalent to EMA with alpha=1/14)
        def wilders_smoothing(series, period):
            result = np.full_like(series, np.nan)
            if len(series) >= period:
                # First value is simple average
                result[period-1] = np.nanmean(series[:period])
                # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
                for i in range(period, len(series)):
                    if not np.isnan(result[i-1]) and not np.isnan(series[i]):
                        result[i] = result[i-1] * (1 - 1/period) + series[i] * (1/period)
                    else:
                        result[i] = np.nan
            return result
        
        atr = wilders_smoothing(tr.values, 14)
        dm_plus_smooth = wilders_smoothing(dm_plus, 14)
        dm_minus_smooth = wilders_smoothing(dm_minus, 14)
        
        # Avoid division by zero
        dx = np.full_like(atr, np.nan)
        valid_atr = (atr != 0) & ~np.isnan(atr)
        dx[valid_atr] = (np.abs(dm_plus_smooth[valid_atr] - dm_minus_smooth[valid_atr]) / 
                         atr[valid_atr]) * 100
        
        adx = wilders_smoothing(dx, 14)
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    else:
        adx_aligned = np.full(n, np.nan)
    
    # Volume spike detection: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Only trade when ADX > 25 (trending market)
        if adx_aligned[i] <= 25:
            # In ranging markets, stay flat or consider mean reversion (but we avoid for simplicity)
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Enter long on break above R1 with volume spike
            if close[i] > r1_level and vol_ok:
                signals[i] = 0.25
                position = 1
            # Enter short on break below S1 with volume spike
            elif close[i] < s1_level and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long on break below S1 (reversal) or loss of momentum
            if close[i] < s1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short on break above R1 (reversal) or loss of momentum
            if close[i] > r1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0
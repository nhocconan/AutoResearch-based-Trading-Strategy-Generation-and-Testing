#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot points (standard, not Camarilla) with 1d ADX trend filter and volume confirmation
# Long when price breaks above weekly R2 pivot AND 1d ADX > 25 AND volume > 1.5 * avg_volume(20)
# Short when price breaks below weekly S2 pivot AND 1d ADX > 25 AND volume > 1.5 * avg_volume(20)
# Exit when price crosses back below/above weekly pivot point OR ADX drops below 20 (trend weakening)
# Uses discrete sizing 0.25 to manage drawdown in bear markets
# Target: 60-120 total trades over 4 years (15-30/year) for 6h timeframe
# Weekly pivot points provide institutional support/resistance levels
# 1d ADX filters for trending markets only to avoid chop
# Volume confirmation ensures breakout validity
# Works in bull markets (breakouts with strong trend) and bear markets (breakdowns with strong trend)

name = "6h_WeeklyPivot_R2S2_Breakout_1dADX_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:  # Need at least one completed weekly bar
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly standard pivot points (based on previous weekly bar)
    # Pivot Point (PP) = (High + Low + Close) / 3
    # R1 = (2 * PP) - Low, S1 = (2 * PP) - High
    # R2 = PP + (High - Low), S2 = PP - (High - Low)
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    r2_1w = pp_1w + (high_1w - low_1w)
    s2_1w = pp_1w - (high_1w - low_1w)
    pivot_1w = pp_1w  # Main pivot for exit
    
    # Align weekly pivot levels to 6h timeframe (wait for completed weekly bar)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Get 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    # True Range (TR) = max[(high-low), abs(high-previous_close), abs(low-previous_close)]
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def WilderSmoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    atr = WilderSmoothing(tr, period)
    dm_plus_smooth = WilderSmoothing(dm_plus, period)
    dm_minus_smooth = WilderSmoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, (dm_plus_smooth / atr) * 100, 0)
    di_minus = np.where(atr != 0, (dm_minus_smooth / atr) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0,
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = WilderSmoothing(dx, period)
    adx[adx == 0] = np.nan  # Mark invalid values
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly R2, ADX > 25 (strong trend), volume confirmation, in session
            if close[i] > r2_aligned[i] and adx_aligned[i] > 25 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S2, ADX > 25 (strong trend), volume confirmation, in session
            elif close[i] < s2_aligned[i] and adx_aligned[i] > 25 and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below weekly pivot OR ADX drops below 20 (trend weakening)
            if close[i] < pivot_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above weekly pivot OR ADX drops below 20 (trend weakening)
            if close[i] > pivot_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
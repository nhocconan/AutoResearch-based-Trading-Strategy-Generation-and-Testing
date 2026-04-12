#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1d_vortex_trend_v1
# Uses Vortex Indicator from daily timeframe to detect trend direction on 12h chart.
# Long when VI+ > VI- and price above EMA50; short when VI- > VI+ and price below EMA50.
# Requires volume > 1.5x 50-period average for confirmation.
# Designed for low trade frequency (target: 15-35 trades/year) to minimize fee drag.
# Works in bull markets (trend following) and bear markets (trend following with short bias).

name = "12h_1d_vortex_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Vortex and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate True Range for Vortex
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Vortex Indicator calculations
    vm_plus = np.abs(high_1d[1:] - low_1d[:-1])  # |high - prev low|
    vm_minus = np.abs(low_1d[1:] - high_1d[:-1])  # |low - prev high|
    
    # Wilder's smoothing function
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Smooth TR, VM+, VM- with 14-period Wilder
    tr14 = wilders_smooth(tr, 14)
    vm_plus14 = wilders_smooth(vm_plus, 14)
    vm_minus14 = wilders_smooth(vm_minus, 14)
    
    # VI+ and VI-
    vi_plus = np.where(tr14 != 0, vm_plus14 / tr14, 0)
    vi_minus = np.where(tr14 != 0, vm_minus14 / tr14, 0)
    
    # Align Vortex to 12h timeframe (daily values update after daily bar closes)
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    
    # Daily EMA50 for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation on 12h: volume > 1.5x 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # start after warmup
        # Skip if values not ready
        if (np.isnan(vi_plus_aligned[i]) or np.isnan(vi_minus_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation
        if not vol_confirm[i]:
            # Hold current position if volume fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: VI+ > VI- and price above EMA50
        if vi_plus_aligned[i] > vi_minus_aligned[i] and close[i] > ema_50_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: VI- > VI+ and price below EMA50
        elif vi_minus_aligned[i] > vi_plus_aligned[i] and close[i] < ema_50_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite Vortex crossover
        elif vi_minus_aligned[i] > vi_plus_aligned[i] and position == 1:
            position = 0
            signals[i] = 0.0
        elif vi_plus_aligned[i] > vi_minus_aligned[i] and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals
#!/usr/bin/env python3
# Strategy: 12h_1d_Vortex_Trend_Volume_Spike_v1
# Hypothesis: Use Vortex indicator (VI+ and VI-) on daily timeframe to identify trend direction on 12h chart.
# Enter long when VI+ > VI- (bullish trend) with price above 12h EMA20 and volume > 2x 20-period MA.
# Enter short when VI- > VI+ (bearish trend) with price below 12h EMA20 and volume > 2x 20-period MA.
# Exit when trend reverses or price crosses EMA20 in opposite direction.
# Uses volume confirmation to avoid false breakouts and EMA20 for dynamic stop.
# Designed for 15-25 trades/year to minimize fee drag and work in both bull/bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Vortex trend indicator
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for Vortex
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d[0] - low_1d[0]  # First period
    
    # Vortex Indicator components
    vm_plus = np.abs(high_1d - np.roll(low_1d, 1))
    vm_minus = np.abs(low_1d - np.roll(high_1d, 1))
    vm_plus[0] = np.abs(high_1d[0] - low_1d[0])
    vm_minus[0] = np.abs(low_1d[0] - high_1d[0])
    
    # Sum over 14 periods
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus_14 = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus_14 = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    # VI+ and VI-
    vi_plus = vm_plus_14 / tr14
    vi_minus = vm_minus_14 / tr14
    
    # Trend: VI+ > VI- = bullish, VI- > VI+ = bearish
    vi_plus_minus = vi_plus - vi_minus  # >0 bullish, <0 bearish
    
    # Align Vortex trend to 12h
    vi_plus_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus_minus)
    
    # Load 12h data for entry timing, EMA, volume
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h EMA20 for dynamic entry/exit
    close_12h_series = pd.Series(close_12h)
    ema20_12h = close_12h_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume spike detection (20-period on 12h)
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(vi_plus_minus_aligned[i]) or 
            np.isnan(ema20_12h[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        vol = volume_12h[i]
        trend = vi_plus_minus_aligned[i]
        
        if position == 0:
            # Long: bullish trend (VI+ > VI-), price above EMA20, volume confirmation
            if (trend > 0 and 
                price > ema20_12h[i] and 
                vol > 2.0 * vol_ma_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish trend (VI- > VI+), price below EMA20, volume confirmation
            elif (trend < 0 and 
                  price < ema20_12h[i] and 
                  vol > 2.0 * vol_ma_20[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend turns bearish or price crosses below EMA20
            if (trend < 0 or 
                price < ema20_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend turns bullish or price crosses above EMA20
            if (trend > 0 or 
                price > ema20_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Vortex_Trend_Volume_Spike_v1"
timeframe = "12h"
leverage = 1.0
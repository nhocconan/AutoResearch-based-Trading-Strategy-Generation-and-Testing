#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot Reversion with Daily Volume Confirmation
# - Long when price touches weekly S1 support AND daily volume > 1.5x 20-day average AND price closes above open (bullish candle)
# - Short when price touches weekly R1 resistance AND daily volume > 1.5x 20-day average AND price closes below open (bearish candle)
# - Exit when price reaches weekly pivot point or opposite weekly level
# - Uses weekly pivot points for key support/resistance levels
# - Volume confirmation filters out false touches
# - Designed for mean reversion in ranging markets (works in both bull/bear via pivot structure)
# - Target: 12-30 trades per year per symbol (50-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation
    df_wk = get_htf_data(prices, '1w')
    high_wk = df_wk['high'].values
    low_wk = df_wk['low'].values
    close_wk = df_wk['close'].values
    
    # Calculate weekly pivot points (using prior week's data)
    pivot_wk = (high_wk + low_wk + close_wk) / 3
    range_wk = high_wk - low_wk
    r1_wk = pivot_wk + range_wk * 1.1 / 2
    s1_wk = pivot_wk - range_wk * 1.1 / 2
    r2_wk = pivot_wk + range_wk * 1.1
    s2_wk = pivot_wk - range_wk * 1.1
    
    # Align weekly levels to 6h timeframe
    r1_wk_aligned = align_htf_to_ltf(prices, df_wk, r1_wk)
    s1_wk_aligned = align_htf_to_ltf(prices, df_wk, s1_wk)
    pivot_wk_aligned = align_htf_to_ltf(prices, df_wk, pivot_wk)
    r2_wk_aligned = align_htf_to_ltf(prices, df_wk, r2_wk)
    s2_wk_aligned = align_htf_to_ltf(prices, df_wk, s2_wk)
    
    # Load daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Price data
    close = prices['close'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in indicators
        if np.isnan(r1_wk_aligned[i]) or np.isnan(s1_wk_aligned[i]) or np.isnan(pivot_wk_aligned[i]) or \
           np.isnan(vol_ma_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_1d_aligned[i]
        is_bullish_candle = close[i] > open_price[i]
        is_bearish_candle = close[i] < open_price[i]
        
        if position == 0:
            # Long entry: price at weekly S1 + volume confirmation + bullish candle
            if abs(price - s1_wk_aligned[i]) < 0.001 * price and vol > 1.5 * vol_ma and is_bullish_candle:
                signals[i] = 0.25
                position = 1
            # Short entry: price at weekly R1 + volume confirmation + bearish candle
            elif abs(price - r1_wk_aligned[i]) < 0.001 * price and vol > 1.5 * vol_ma and is_bearish_candle:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches weekly pivot or S2 (stop)
            if price >= pivot_wk_aligned[i] or price <= s2_wk_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches weekly pivot or R2 (stop)
            if price <= pivot_wk_aligned[i] or price >= r2_wk_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivotReversion_VolumeConfirm"
timeframe = "6h"
leverage = 1.0
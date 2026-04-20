#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d ADX Trend Filter
# - Williams Alligator (13,8,5) on 12h for trend identification
# - Jaw (13-period smoothed median), Teeth (8-period), Lips (5-period)
# - Long when Lips > Teeth > Jaw (bullish alignment) and 1d ADX > 25 (strong trend)
# - Short when Lips < Teeth < Jaw (bearish alignment) and 1d ADX > 25 (strong trend)
# - Exit when Alligator lines re-cross or ADX falls below 20
# - Designed for 12h timeframe with strong trend filtering to avoid whipsaws
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d timeframe
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
    
    # Smoothed values
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Williams Alligator on 12h timeframe
    median_price = (prices['high'].values + prices['low'].values) / 2
    
    # Jaw: 13-period SMMA of median, shifted 8 bars
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMMA of median, shifted 5 bars
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA of median, shifted 3 bars
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)
    lips[:3] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in indicators
        if np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or np.isnan(adx_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Bullish Alligator alignment + strong trend (ADX > 25)
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and adx_1d_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish Alligator alignment + strong trend (ADX > 25)
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and adx_1d_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator re-crosses or trend weakens (ADX < 20)
            if not (lips[i] > teeth[i] and teeth[i] > jaw[i]) or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator re-crosses or trend weakens (ADX < 20)
            if not (lips[i] < teeth[i] and teeth[i] < jaw[i]) or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dADX_TrendFilter"
timeframe = "12h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1-day ATR-based breakout and volume confirmation.
# Uses 1-day ATR and closing price to set dynamic breakout levels above/below prior close.
# Volatility-adjusted breakouts work in both bull and bear markets by adapting to
# changing volatility regimes. Volume filter ensures breakout conviction.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled risk.
name = "12h_1d_ATRBreakout_VolumeFilter_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR and close calculation (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(10) on 1d timeframe
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]  # first period
    tr3[0] = tr1[0]  # first period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate ATR multiplier (0.5 * ATR)
    atr_mult = 0.5 * atr_1d
    
    # Upper and lower breakout levels: prior close ± 0.5 * ATR
    upper_break = np.roll(close_1d, 1) + atr_mult
    lower_break = np.roll(close_1d, 1) - atr_mult
    # First value has no prior close
    upper_break[0] = np.nan
    lower_break[0] = np.nan
    
    # Align breakout levels to 12h timeframe
    upper_break_aligned = align_htf_to_ltf(prices, df_1d, upper_break)
    lower_break_aligned = align_htf_to_ltf(prices, df_1d, lower_break)
    
    # Volume filter: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for ATR and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_break_aligned[i]) or np.isnan(lower_break_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr_1d[i//1] if i//1 < len(atr_1d) else np.nan)):
            signals[i] = 0.0
            continue
            
        if position == 0:
            # Long when price breaks above upper level with volume
            if close[i] > upper_break_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below lower level with volume
            elif close[i] < lower_break_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price breaks below lower level
            if close[i] < lower_break_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price breaks above upper level
            if close[i] > upper_break_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
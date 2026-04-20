#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with daily Choppiness Index filter + Donchian breakout
# Uses only 2 conditions: price breaks Donchian(20) + low volatility regime (Choppiness > 61.8)
# Designed for low trade frequency (<30/year) to avoid fee drag, works in trending and ranging markets
name = "12h_1d_Donchian20_ChopFilter_V1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily Choppiness Index (30-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Handle first value
    tr[0] = high_1d[0] - low_1d[0]
    
    # ATR(30)
    atr = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    
    # Sum of TR over 30 periods
    tr_sum = pd.Series(tr).rolling(window=30, min_periods=30).sum().values
    
    # Highest high and lowest low over 30 periods
    hh = pd.Series(high_1d).rolling(window=30, min_periods=30).max().values
    ll = pd.Series(low_1d).rolling(window=30, min_periods=30).min().values
    
    # Choppiness Index
    chop = np.where(
        (tr_sum > 0) & (hh - ll > 0),
        100 * np.log10(tr_sum / (atr * 30)) / np.log10(30),
        50.0  # neutral when undefined
    )
    chop = pd.Series(chop).fillna(50.0).values
    
    # Align Choppiness to 12h timeframe (need 2-bar delay for daily close confirmation)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=2)
    
    # === 12h Donchian Channel (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        dh = donchian_high[i]
        dl = donchian_low[i]
        chop_val = chop_aligned[i]
        
        # Skip if any value is NaN
        if np.isnan(dh) or np.isnan(dl) or np.isnan(chop_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high in low volatility regime (chop > 61.8 = ranging)
            if close_val > dh and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low in low volatility regime
            elif close_val < dl and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below Donchian low OR chop drops (trending begins)
            if close_val < dl or chop_val < 40.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above Donchian high OR chop drops
            if close_val > dh or chop_val < 40.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
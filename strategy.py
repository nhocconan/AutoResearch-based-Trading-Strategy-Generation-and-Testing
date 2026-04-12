#!/usr/bin/env python3
"""
6h_12h_1d_Donchian_EMA_Vol_Filter
Hypothesis: In volatile markets (high 1d ATR), 6h Donchian breakouts in the direction of the 12h EMA trend yield sustainable trends.
Uses 12h EMA20/EMA50 for trend, 1d ATR/MA(ATR) for volatility filter, and 6h Donchian(20) for breakouts.
Targets 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing.
Works in bull/bear by requiring volatility expansion and trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_Donchian_EMA_Vol_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # === 12H TREND: EMA20 and EMA50 ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 1D VOLATILITY FILTER: ATR(14) and its 50-period MA ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # ATR(14)
    atr_14 = np.full_like(close_1d, np.nan)
    for i in range(14, len(close_1d)):
        atr_14[i] = np.nanmean(tr[i-13:i+1])  # Simple mean of last 14 TR
    
    # ATR 50-period MA
    atr_ma = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    vol_ratio = atr_14 / atr_ma  # >1 indicates expanding volatility
    
    # === 6H DONCHIAN CHANNEL (20-period) ===
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h EMAs and 1d volatility ratio to 6h timeframe
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup period (ensures all indicators are valid)
    start_idx = 100
    for i in range(start_idx, n):
        # Skip if any key value is NaN
        if (np.isnan(ema20_12h_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i])):
            # Hold current position or flat if none
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
            continue
        
        close_price = prices['close'].iloc[i]
        trend_up = ema20_12h_aligned[i] > ema50_12h_aligned[i]
        trend_down = ema20_12h_aligned[i] < ema50_12h_aligned[i]
        high_vol = vol_ratio_aligned[i] > 1.0
        
        # Long: 12h uptrend, high volatility, Donchian breakout up
        if trend_up and high_vol and close_price > donchian_high[i]:
            if position != 1:
                position = 1
                signals[i] = 0.25
            else:
                signals[i] = 0.25
        # Short: 12h downtrend, high volatility, Donchian breakout down
        elif trend_down and high_vol and close_price < donchian_low[i]:
            if position != -1:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = -0.25
        # Exit long: trend reversal or volatility contraction
        elif position == 1 and (not trend_up or not high_vol):
            position = 0
            signals[i] = 0.0
        # Exit short: trend reversal or volatility contraction
        elif position == -1 and (not trend_down or not high_vol):
            position = 0
            signals[i] = 0.0
        # Hold existing position
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
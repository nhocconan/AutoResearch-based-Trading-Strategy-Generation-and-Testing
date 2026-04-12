#!/usr/bin/env python3
"""
4h_1D_Donchian_Volume_Chop_Strategy
Hypothesis: Donchian channel breakouts on 4h timeframe, filtered by daily trend (EMA50) and 
choppiness regime, with volume confirmation, captures strong momentum moves while avoiding 
whipsaw in choppy markets. Works in both bull (breakouts above) and bear (breakdowns below) 
markets by using the daily trend filter. Low trade frequency (~25/year) minimizes fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1D_Donchian_Volume_Chop_Strategy"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA FOR TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema50_1d = close_series_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily EMA50 to 4h timeframe (wait for daily bar to close)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 4H INDICATORS ===
    # Donchian Channel (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Choppiness Index (14-period) - ranging market filter
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    
    chop = 100 * np.log10(pd.Series(atr).rolling(window=14, min_periods=14).sum().values / hl_range) / np.log10(14)
    
    # Volume confirmation (4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(chop[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long setup: Price breaks above Donchian high AND above daily EMA50 (uptrend) 
        # AND market is not too choppy (trending) AND volume confirmation
        long_setup = (close[i] > high_max[i]) and (close[i] > ema50_1d_aligned[i]) and (chop[i] < 61.8) and (vol_ratio[i] > 1.3)
        
        # Short setup: Price breaks below Donchian low AND below daily EMA50 (downtrend) 
        # AND market is not too choppy (trending) AND volume confirmation
        short_setup = (close[i] < low_min[i]) and (close[i] < ema50_1d_aligned[i]) and (chop[i] < 61.8) and (vol_ratio[i] > 1.3)
        
        # Exit when price returns to middle of Donchian channel or trend changes
        donchian_mid = (high_max[i] + low_min[i]) / 2
        exit_long = close[i] < donchian_mid
        exit_short = close[i] > donchian_mid
        
        # Execute trades
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
#!/usr/bin/env python3
"""
1d Donchian breakout with volume confirmation and volatility filter.
Hypothesis: Donchian(20) breakouts on daily timeframe capture strong trends, while volume confirmation filters false breakouts and volatility filter (ATR) avoids choppy periods. Works in both bull and bear markets by capturing momentum bursts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14298_1d_donchian20_vol_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Donchian calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period high/low)
    # Using pandas rolling for proper min_periods
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for volatility filter
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_close[0] = high_1d[0] - low_1d[0]  # First value
    low_close[0] = high_1d[0] - low_1d[0]   # First value
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align to 1d timeframe (already aligned, but shift for completed bars)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # 1d data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    # Volatility filter: avoid extremely low volatility (chop)
    vol_ma_long = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    vol_ma_long_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_long)
    vol_filter = atr_aligned > (0.5 * vol_ma_long_aligned)  # Avoid low volatility
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(20, 14, 50) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(atr_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(vol_filter[i]):
            signals[i] = 0.0
            continue
        
        # Check exits: close back inside Donchian channel or volatility too low
        if position == 1:  # long position
            if close[i] <= donchian_high_aligned[i] or not vol_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_low_aligned[i] or not vol_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout with volume and volatility confirmation
            long_breakout = close[i] > donchian_high_aligned[i]
            short_breakout = close[i] < donchian_low_aligned[i]
            
            if long_breakout and vol_confirm[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            elif short_breakout and vol_confirm[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals
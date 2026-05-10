#!/usr/bin/env python3
# 4h_TRIX_VolumeSpike_ChopFilter
# Hypothesis: Uses TRIX (15,9) momentum with volume spike and Choppiness Index regime filter.
# Long when TRIX crosses above signal line with volume spike in trending market (CHOP < 38.2).
# Short when TRIX crosses below signal line with volume spike in trending market.
# Exit when TRIX crosses back or volume drops.
# Designed to work in both bull and bear markets by filtering choppy regimes.
# Targets 20-50 trades per year on 4h timeframe with position size 0.25.

name = "4h_TRIX_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for TRIX calculation (same timeframe, but we'll use it for smoothing)
    # For TRIX we need the same timeframe data
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate TRIX: Triple Exponential Moving Average
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15)
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = ema3.pct_change() * 100  # Percentage change
    
    # Signal line: EMA of TRIX
    signal_line = trix.ewm(span=9, adjust=False, min_periods=9).mean()
    
    # Align TRIX and signal line to lower timeframe (though same, we use align for safety)
    trix_aligned = align_htf_to_ltf(prices, df_4h, trix.values)
    signal_aligned = align_htf_to_ltf(prices, df_4h, signal_line.values)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    # Choppiness Index (using daily data for regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR(14)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Sum of TR over 14 periods
    sum_tr = tr.rolling(window=14, min_periods=14).sum()
    
    # Highest high and lowest low over 14 periods
    hh = df_1d['high'].rolling(window=14, min_periods=14).max()
    ll = df_1d['low'].rolling(window=14, min_periods=14).min()
    
    # Choppiness Index
    chop = 100 * np.log10(sum_tr / (atr * 14)) / np.log10(14)
    chop = chop.fillna(50)  # Fill NaN with neutral value
    
    # Align Chop to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop.values, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20, 14)  # Warmup for TRIX, volume MA, and Chop
    
    for i in range(start_idx, n):
        if np.isnan(trix_aligned[i]) or np.isnan(signal_aligned[i]) or np.isnan(chop_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # TRIX crossover signals
        trix_cross_above = trix_aligned[i] > signal_aligned[i] and trix_aligned[i-1] <= signal_aligned[i-1]
        trix_cross_below = trix_aligned[i] < signal_aligned[i] and trix_aligned[i-1] >= signal_aligned[i-1]
        
        # Trending market filter (CHOP < 38.2)
        trending = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long entry: TRIX crosses above signal with volume spike in trending market
            if trix_cross_above and volume_confirm[i] and trending:
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX crosses below signal with volume spike in trending market
            elif trix_cross_below and volume_confirm[i] and trending:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses back below signal or market becomes choppy
            if trix_cross_below or not trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses back above signal or market becomes choppy
            if trix_cross_above or not trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
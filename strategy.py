#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1-day and 1-week trend filters.
Trade 4h breakouts of Donchian channels (20-period) with 1-day EMA200 and 1-week EMA50 trend filters,
plus volume confirmation and a tight exit at the Donchian midpoint to avoid whipsaw.
Use only 4h candles for signals to maintain low trade frequency (~25-40 per year).
Designed to work in bull markets via trend-following breakouts and in bear via mean-reversion at structure.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for structure (Donchian channels)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_max_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_max_20 + low_min_20) / 2.0  # midpoint for exit
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(200) for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h, 1d, and 1w data to 4h
    high_max_20_aligned = align_htf_to_ltf(prices, df_4h, high_max_20)
    low_min_20_aligned = align_htf_to_ltf(prices, df_4h, low_min_20)
    mid_20_aligned = align_htf_to_ltf(prices, df_4h, mid_20)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: current volume > 1.5x 24-period average (to avoid noise)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max_20_aligned[i]) or np.isnan(low_min_20_aligned[i]) or 
            np.isnan(mid_20_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian high with volume and above both EMAs
            if close[i] > high_max_20_aligned[i] and volume_filter[i] and \
               close[i] > ema_200_1d_aligned[i] and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 4h Donchian low with volume and below both EMAs
            elif close[i] < low_min_20_aligned[i] and volume_filter[i] and \
                 close[i] < ema_200_1d_aligned[i] and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below the 4h Donchian midpoint (mean reversion)
            if close[i] < mid_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above the 4h Donchian midpoint (mean reversion)
            if close[i] > mid_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA200_1wEMA50_Volume_MidExit"
timeframe = "4h"
leverage = 1.0
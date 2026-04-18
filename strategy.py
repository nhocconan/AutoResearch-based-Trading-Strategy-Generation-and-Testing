# 12h_Pivot_R1S1_Breakout_Volume_1dTrendFilter_V1
# Hypothesis: Breakouts from Camarilla pivot levels (R1, S1) on 12h timeframe with volume confirmation
# and 1d trend filter (price > EMA200 for long, price < EMA200 for short). Camarilla levels
# derived from prior 1d candle. Breakouts occur when price pierces these levels with momentum.
# Volume spike confirms institutional participation. EMA200 filter ensures alignment with
# daily trend to avoid counter-trend whipsaws. Designed for low trade frequency (target: 12-37/year)
# to minimize fee drag while capturing significant moves in both bull and bear markets.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Pivot_R1S1_Breakout_Volume_1dTrendFilter_V1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from prior 1d candle
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_range = high_1d - low_1d
    r1 = close_1d + (1.1 * pivot_range / 12)
    s1 = close_1d - (1.1 * pivot_range / 12)
    
    # Align pivot levels to 12h timeframe (use prior day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate EMA200 on 1d for trend filter
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for EMA200 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema200_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume spike AND price > EMA200 (uptrend)
            if close[i] > r1_aligned[i] and volume_spike[i] and close[i] > ema200_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike AND price < EMA200 (downtrend)
            elif close[i] < s1_aligned[i] and volume_spike[i] and close[i] < ema200_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below R1 OR trend reverses (price < EMA200)
            if close[i] < r1_aligned[i] or close[i] < ema200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above S1 OR trend reverses (price > EMA200)
            if close[i] > s1_aligned[i] or close[i] > ema200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
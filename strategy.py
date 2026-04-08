#!/usr/bin/env python3

# 6h_ema200_breakout_volume_regime_v2
# Hypothesis: Trend-following strategy using EMA200 from 1d timeframe as trend filter and EMA50 from 6t for momentum.
# Enters on EMA50/EMA200 crossovers with volume confirmation and exits on opposite crossover.
# Designed to work in both bull and bear markets by following the dominant trend on higher timeframe.
# Target: 12-37 trades per year (50-150 total over 4 years) for low fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ema200_breakout_volume_regime_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily trend filter (1d EMA200) - load once before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily data
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 6h indicators
    # EMA50 for momentum
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 200  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(ema50[i]) or np.isnan(ema200_1d_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Daily trend filter
        daily_uptrend = close[i] > ema200_1d_aligned[i]
        daily_downtrend = close[i] < ema200_1d_aligned[i]
        
        # EMA crossover signals
        ema50_above_200 = ema50[i] > ema200_1d_aligned[i]
        ema50_below_200 = ema50[i] < ema200_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.3 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit when EMA50 crosses below EMA200 (trend change)
            if ema50_below_200:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when EMA50 crosses above EMA200 (trend change)
            if ema50_above_200:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            if volume_ok:
                # Long entry: EMA50 crosses above EMA200 in uptrend
                if daily_uptrend and ema50_above_200 and ema50[i-1] <= ema200_1d_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: EMA50 crosses below EMA200 in downtrend
                elif daily_downtrend and ema50_below_200 and ema50[i-1] >= ema200_1d_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
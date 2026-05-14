#!/usr/bin/env python3
"""
4h_Camarilla_R2_S2_Breakout_1dEMA34_Trend_Volume
Hypothesis: Moderate breakouts at R2/S2 levels with volume confirmation and 1d EMA34 trend filter capture institutional moves while avoiding false breakouts. Works in bull (breakouts continue) and bear (breakdowns continue) markets. Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_cam = df_1d['close'].values
    
    # Camarilla R2, S2 levels: H/L from previous day
    R2 = close_1d_cam + (high_1d - low_1d) * 1.1 / 4
    S2 = close_1d_cam - (high_1d - low_1d) * 1.1 / 4
    
    # Align to 4h timeframe (previous day's levels available at open)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 20  # need 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(R2_aligned[i]) or 
            np.isnan(S2_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R2 with volume spike and uptrend on 1d
            if (close[i] > R2_aligned[i] and volume_spike[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 with volume spike and downtrend on 1d
            elif (close[i] < S2_aligned[i] and volume_spike[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns below S2 or trend fails
            if (close[i] < S2_aligned[i] or 
                close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns above R2 or trend fails
            if (close[i] > R2_aligned[i] or 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R2_S2_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0
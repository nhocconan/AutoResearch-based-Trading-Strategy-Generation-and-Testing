#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla pivot levels from 1d are strong support/resistance. Breakout above R1 or below S1 with 1d trend filter and volume confirmation. Uses 12h timeframe to reduce trade frequency and avoid fee drag. Works in bull markets by buying breakouts above R1 in uptrends, and in bear markets by selling breakdowns below S1 in downtrends. Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

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
    
    # 1d data for Camarilla pivot levels, trend filter, and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Camarilla pivot levels: R1, S1 from previous day
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    camarilla_range = high_1d - low_1d
    r1 = close_1d + 1.1 * camarilla_range / 12
    s1 = close_1d - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d volume SMA20 for volume confirmation
    vol_sma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA34 to be ready
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or \
           np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 1d volume (scaled)
        vol_1d_scaled = vol_sma20_1d_aligned[i] / 2.0  # 2x 12h periods in 1d
        volume_confirm = volume[i] > 1.5 * vol_1d_scaled
        
        # Trend determination: price vs 1d EMA34
        is_uptrend = close[i] > ema34_1d_aligned[i]
        is_downtrend = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # Long: Breakout above R1 in uptrend with volume
            if (close[i] > r1_aligned[i] and
                is_uptrend and
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below S1 in downtrend with volume
            elif (close[i] < s1_aligned[i] and
                  is_downtrend and
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price returns below R1 or trend changes to downtrend
            if (close[i] < r1_aligned[i] or
                not is_uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price returns above S1 or trend changes to uptrend
            if (close[i] > s1_aligned[i] or
                not is_downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
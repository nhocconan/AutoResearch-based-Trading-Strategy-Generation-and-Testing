#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla pivot levels from 1d timeframe act as strong support/resistance on 6h.
Breakout above R3 or below S3 with 1d trend alignment (EMA34) and volume spike (>2x 20-period MA) 
indicates institutional participation and continuation. Fade at R4/S4 with contrarian volume.
Designed to work in both bull and bear markets by following 1d trend for breakouts and 
fading extremes in ranging conditions. Target: 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    #           S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's levels (no look-ahead)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    # Calculate Camarilla levels
    R4 = close_1d_prev + 1.5 * (high_1d_prev - low_1d_prev)
    R3 = close_1d_prev + 1.1 * (high_1d_prev - low_1d_prev)
    S3 = close_1d_prev - 1.1 * (high_1d_prev - low_1d_prev)
    S4 = close_1d_prev - 1.5 * (high_1d_prev - low_1d_prev)
    
    # Align to 6h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    uptrend_1d = close > ema_34_1d_aligned
    downtrend_1d = close < ema_34_1d_aligned
    
    # Volume confirmation: volume > 2.0x 20-period MA (stricter for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 35 for roll + 34 for EMA + 20 for volume MA)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long breakout: price > R3 with uptrend and volume spike
            if (close[i] > R3_aligned[i] and 
                uptrend_1d[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: price < S3 with downtrend and volume spike
            elif (close[i] < S3_aligned[i] and 
                  downtrend_1d[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            # Contrarian fade at extremes: price > R4 in downtrend OR price < S4 in uptrend
            elif (close[i] > R4_aligned[i] and downtrend_1d[i]) or \
                 (close[i] < S4_aligned[i] and uptrend_1d[i]):
                # Fade the extreme move
                if close[i] > R4_aligned[i] and downtrend_1d[i]:
                    signals[i] = -0.25  # Short the overbought extreme
                    position = -1
                else:
                    signals[i] = 0.25   # Long the oversold extreme
                    position = 1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price drops below R3 (breakout failed) OR 1d trend changes to downtrend
            if (close[i] < R3_aligned[i]) or (not uptrend_1d[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above S3 (breakdown failed) OR 1d trend changes to uptrend
            if (close[i] > S3_aligned[i]) or (not downtrend_1d[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0
# 12H_Camarilla_R1S1_1DTrend_VolumeBreakout
# Hypothesis: For 12h timeframe, use weekly trend filter (price above/below weekly SMA50) with daily Camarilla R1/S1 breakout and volume confirmation.
# Weekly trend reduces false signals in choppy markets, volume confirms breakout strength.
# Target: 15-25 trades/year per symbol, suitable for 12h timeframe.
# Works in bull markets (trend-following breakouts) and bear markets (counter-trend reversals at S1 in downtrend).

name = "12H_Camarilla_R1S1_1DTrend_VolumeBreakout"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla R1 and S1 levels from previous daily period's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    hl_range = high_1d - low_1d
    r1_1d = close_1d + 1.1 * hl_range / 4
    s1_1d = close_1d - 1.1 * hl_range / 4
    
    # Align R1/S1 to 12h timeframe (use previous daily period's levels)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate weekly SMA50 for trend filter
    sma50_1w = pd.Series(df_1w['close'].values).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    # Volume spike detection: 1.5x average volume (20-period for 12h responsiveness)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure we have volume MA and weekly SMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(sma50_1w_aligned[i]) or np.isnan(vol_ma[i]) or
            vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above daily R1, price above weekly SMA50 (uptrend), volume spike
            if (close[i] > r1_1d_aligned[i] and 
                close[i] > sma50_1w_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily S1, price below weekly SMA50 (downtrend), volume spike
            elif (close[i] < s1_1d_aligned[i] and 
                  close[i] < sma50_1w_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to or below daily S1 (opposite level)
            if close[i] <= s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to or above daily R1 (opposite level)
            if close[i] >= r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
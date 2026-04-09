#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d ADX regime filter
# Williams %R(14) identifies overbought/oversold conditions
# Enters long when %R < -80 (oversold) and 1d ADX < 25 (low trend = mean reversion regime)
# Enters short when %R > -20 (overbought) and 1d ADX < 25
# Exits when %R crosses above -50 (for longs) or below -50 (for shorts)
# Position size 0.25 to limit drawdown
# Target: 50-150 total trades over 4 years (~12-37/year) to minimize fee drag
# Works in both bull/bear markets by focusing on mean reversion in low-trend regimes

name = "6h_1d_williamsr_adx_meanrev_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr_1d = np.zeros(len(df_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr0 = high_1d[i] - low_1d[i]
        tr1 = abs(high_1d[i] - close_1d[i-1])
        tr2 = abs(low_1d[i] - close_1d[i-1])
        tr_1d[i] = max(tr0, tr1, tr2)
    
    # Directional Movement
    dm_plus_1d = np.zeros(len(df_1d))
    dm_minus_1d = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        dm_plus_1d[i] = up_move if up_move > down_move and up_move > 0 else 0
        dm_minus_1d[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = alpha = 1/14)
    tr_14_1d = np.zeros(len(df_1d))
    dm_plus_14_1d = np.zeros(len(df_1d))
    dm_minus_14_1d = np.zeros(len(df_1d))
    
    # Initial values (first 14 periods)
    if len(df_1d) >= 14:
        tr_14_1d[13] = np.sum(tr_1d[0:14])
        dm_plus_14_1d[13] = np.sum(dm_plus_1d[0:14])
        dm_minus_14_1d[13] = np.sum(dm_minus_1d[0:14])
        
        # Wilder's smoothing for remaining periods
        for i in range(14, len(df_1d)):
            tr_14_1d[i] = tr_14_1d[i-1] - (tr_14_1d[i-1] / 14) + tr_1d[i]
            dm_plus_14_1d[i] = dm_plus_14_1d[i-1] - (dm_plus_14_1d[i-1] / 14) + dm_plus_1d[i]
            dm_minus_14_1d[i] = dm_minus_14_1d[i-1] - (dm_minus_14_1d[i-1] / 14) + dm_minus_1d[i]
    
    # Directional Indicators
    di_plus_1d = np.zeros(len(df_1d))
    di_minus_1d = np.zeros(len(df_1d))
    dx_1d = np.zeros(len(df_1d))
    
    for i in range(14, len(df_1d)):
        if tr_14_1d[i] != 0:
            di_plus_1d[i] = (dm_plus_14_1d[i] / tr_14_1d[i]) * 100
            di_minus_1d[i] = (dm_minus_14_1d[i] / tr_14_1d[i]) * 100
            di_sum = di_plus_1d[i] + di_minus_1d[i]
            if di_sum != 0:
                dx_1d[i] = abs(di_plus_1d[i] - di_minus_1d[i]) / di_sum * 100
    
    # ADX (smoothed DX)
    adx_1d = np.zeros(len(df_1d))
    if len(df_1d) >= 28:  # Need 14 for DX + 14 for smoothing
        adx_1d[27] = np.mean(dx_1d[14:28])  # First ADX is average of first 14 DX values
        for i in range(28, len(df_1d)):
            adx_1d[i] = (adx_1d[i-1] * 13 + dx_1d[i]) / 14  # Wilder's smoothing
    
    # Align 1d ADX to 6h timeframe (only use completed daily bars)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Williams %R on 6h (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = np.full(n, np.nan)
    lowest_low_14 = np.full(n, np.nan)
    williams_r = np.full(n, np.nan)
    
    for i in range(13, n):  # Start at index 13 for 14-period lookback
        highest_high_14[i] = np.max(high[i-13:i+1])
        lowest_low_14[i] = np.min(low[i-13:i+1])
        hh_ll = highest_high_14[i] - lowest_low_14[i]
        if hh_ll != 0:
            williams_r[i] = (highest_high_14[i] - close[i]) / hh_ll * -100
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(28, n):  # Start after ADX warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in low trend environment (ADX < 25 = weak trend = mean reversion regime)
        if adx_1d_aligned[i] >= 25:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses above -50 (recovery from oversold)
            if williams_r[i] >= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -50 (decline from overbought)
            if williams_r[i] <= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: Williams %R < -80 (oversold) in low ADX regime
            if williams_r[i] < -80:
                position = 1
                signals[i] = 0.25
            # Enter short: Williams %R > -20 (overbought) in low ADX regime
            elif williams_r[i] > -20:
                position = -1
                signals[i] = -0.25
    
    return signals
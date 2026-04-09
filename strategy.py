#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extremes + 1w trend filter + volume confirmation
# Williams %R > -20 = overbought (short), < -80 = oversold (long) on 1d timeframe
# 1w EMA(34) determines trend: only take longs above EMA, shorts below EMA
# Volume confirmation: current 6h volume > 1.5x 20-period average
# Works in bull/bear markets: mean reversion in ranging phases, trend filter avoids counter-trend trades
# Target: 12-30 trades/year (50-120 total over 4 years) with discrete sizing 0.25 to minimize fee drag

name = "6h_1d_1w_williamsr_extreme_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14-period)
    highest_high_1d = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    williams_r_1d = -100 * (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d)
    
    # Calculate 1w EMA(34) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d Williams %R and 1w EMA to 6h timeframe
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_1d_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit long if Williams %R rises above -50 (exiting overbought territory)
            if williams_r_1d_aligned[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if Williams %R falls above -50 (exiting oversold territory)
            if williams_r_1d_aligned[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion entries with trend filter and volume confirmation
            if (williams_r_1d_aligned[i] < -80 and  # Oversold
                close[i] > ema_34_1w_aligned[i] and  # Above 1w EMA (uptrend)
                volume_confirmed):
                position = 1
                signals[i] = 0.25
            elif (williams_r_1d_aligned[i] > -20 and  # Overbought
                  close[i] < ema_34_1w_aligned[i] and  # Below 1w EMA (downtrend)
                  volume_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals
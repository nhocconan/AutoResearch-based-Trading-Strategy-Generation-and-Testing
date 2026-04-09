#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extreme + 1w EMA trend filter + volume confirmation
# Williams %R(14) on 1d identifies overbought/oversold conditions with mean reversion edge
# 1w EMA(50) provides major trend filter: only long when price > EMA50, short when price < EMA50
# Volume confirmation (current 6h volume > 2.0x 20-period average) filters low-momentum signals
# Designed for 6h timeframe targeting 15-25 trades/year (60-100 over 4 years)
# Works in bull/bear: mean reversion in ranges, trend filter prevents counter-trend in strong moves

name = "6h_1d_1w_williamsr_extreme_v1"
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
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1d Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high_14 - lowest_low_14) != 0,
        ((highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)) * -100,
        0.0
    )
    
    # Calculate 1w EMA(50)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d Williams %R and 1w EMA to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 2.0x average 6h volume
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit conditions: Williams %R > -20 (overbought) OR loss of volume confirmation
            if williams_r_aligned[i] > -20.0 or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: Williams %R < -80 (oversold) OR loss of volume confirmation
            if williams_r_aligned[i] < -80.0 or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion entries with volume confirmation and trend filter
            # Long: Williams %R < -80 (oversold) AND price > 1w EMA50 (uptrend filter)
            # Short: Williams %R > -20 (overbought) AND price < 1w EMA50 (downtrend filter)
            if volume_confirmed:
                if williams_r_aligned[i] < -80.0 and close[i] > ema_50_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif williams_r_aligned[i] > -20.0 and close[i] < ema_50_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
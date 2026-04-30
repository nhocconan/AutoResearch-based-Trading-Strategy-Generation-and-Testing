#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d Supertrend trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold), 1d Supertrend uptrend, and volume > 1.5x 20-bar avg.
# Short when Williams %R > -20 (overbought), 1d Supertrend downtrend, and volume > 1.5x 20-bar avg.
# Exit on opposite Williams %R level (%R > -50 for long exit, %R < -50 for short exit).
# Williams %R identifies overextended moves, Supertrend filters for trend direction,
# volume confirmation reduces false signals. Designed to work in both bull and bear markets
# by capturing mean reversion within the prevailing trend.
# Timeframe: 12h, HTF: 1d as per experiment guidelines.

name = "12h_WilliamsR_MeanRev_1dSupertrend_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d Supertrend for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ATR calculation
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d)
    tr2 = pd.Series(high_1d).shift(1) - pd.Series(close_1d)
    tr3 = pd.Series(close_1d).shift(1) - pd.Series(low_1d)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2_1d = (high_1d + low_1d) / 2
    upper_band_1d = hl2_1d + (3.0 * atr_1d)
    lower_band_1d = hl2_1d - (3.0 * atr_1d)
    
    supertrend_1d = np.full_like(close_1d, np.nan, dtype=float)
    direction_1d = np.full_like(close_1d, np.nan, dtype=float)
    
    for i in range(1, len(close_1d)):
        if np.isnan(supertrend_1d[i-1]):
            # Initialize
            supertrend_1d[i] = lower_band_1d[i]
            direction_1d[i] = 1
        else:
            if close_1d[i] <= supertrend_1d[i-1]:
                supertrend_1d[i] = upper_band_1d[i]
                direction_1d[i] = -1
            else:
                supertrend_1d[i] = lower_band_1d[i]
                direction_1d[i] = 1
                
            # Adjust bands
            if direction_1d[i] == direction_1d[i-1]:
                if direction_1d[i] == 1 and lower_band_1d[i] < supertrend_1d[i-1]:
                    supertrend_1d[i] = supertrend_1d[i-1]
                elif direction_1d[i] == -1 and upper_band_1d[i] > supertrend_1d[i-1]:
                    supertrend_1d[i] = supertrend_1d[i-1]
    
    # Align Supertrend direction to 12h timeframe (wait for 1d bar to close)
    supertrend_1d_aligned = align_htf_to_ltf(prices, df_1d, direction_1d)
    
    # Calculate Williams %R on 12h data
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, lookback)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(supertrend_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_supertrend = supertrend_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: oversold (%R < -80), uptrend (Supertrend > 0), volume spike
            if (curr_williams_r < -80 and 
                curr_supertrend > 0 and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: overbought (%R > -20), downtrend (Supertrend < 0), volume spike
            elif (curr_williams_r > -20 and 
                  curr_supertrend < 0 and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: Williams %R returns above -50 (mean reversion)
            if curr_williams_r > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: Williams %R returns below -50 (mean reversion)
            if curr_williams_r < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
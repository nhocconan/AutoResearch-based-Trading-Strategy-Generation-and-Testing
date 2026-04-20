#!/usr/bin/env python3
# 1h_4h_1d_TrendFollowing_VolumeBreakout
# Hypothesis: 1h breakout with 4h trend filter and 1d volume confirmation. 
# Long when price breaks above 4h high with rising volume and 1d uptrend.
# Short when price breaks below 4h low with rising volume and 1d downtrend.
# Session filter (08-20 UTC) reduces noise. Target: 15-37 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_TrendFollowing_VolumeBreakout"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data ONCE before loop for trend and breakout levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 4h: Highest high and lowest low of last 20 periods ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # === 4h: Trend filter using EMA34 ===
    close_4h = df_4h['close'].values
    ema_34 = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === 1d: Volume ratio (current vs 20-period average) ===
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / np.where(vol_ma20_1d > 0, vol_ma20_1d, np.nan)
    
    # === Align all HTF data to 1h timeframe ===
    high_20_aligned = align_htf_to_ltf(prices, df_4h, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_4h, low_20)
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA warmup
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = prices['close'].iloc[i]
        high_20_val = high_20_aligned[i]
        low_20_val = low_20_aligned[i]
        ema_34_val = ema_34_aligned[i]
        vol_ratio_val = vol_ratio_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(high_20_val) or np.isnan(low_20_val) or 
            np.isnan(ema_34_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above 4h high with volume confirmation and 1d uptrend
            if (close_val > high_20_val and  # Break above 20-period high
                vol_ratio_val > 1.5 and      # Volume confirmation
                close_val > ema_34_val):     # Above 4h EMA34 (uptrend)
                signals[i] = 0.20
                position = 1
            # Short: Break below 4h low with volume confirmation and 1d downtrend
            elif (close_val < low_20_val and   # Break below 20-period low
                  vol_ratio_val > 1.5 and      # Volume confirmation
                  close_val < ema_34_val):     # Below 4h EMA34 (downtrend)
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: Break below 4h low or loss of momentum
            if close_val < low_20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Break above 4h high or loss of momentum
            if close_val > high_20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
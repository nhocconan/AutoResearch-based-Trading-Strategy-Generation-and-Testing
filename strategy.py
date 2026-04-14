#!/usr/bin/env python3
"""
1d_Wide_Range_Breakout
Hypothesis: On the daily timeframe, breakouts from wide daily ranges (high-low > 1.5x ATR) 
with volume confirmation and weekly trend filter capture strong momentum moves. 
Wide range indicates institutional participation; breakouts in trend direction have higher 
follow-through. Works in bull (upside breakouts) and bear (downside breakouts) via symmetry.
Target: 20-50 trades over 4 years (5-12/year) to minimize fee drag.
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
    
    # Load daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate daily true range and ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range components
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily range (high - low)
    daily_range = high_1d - low_1d
    
    # Wide range condition: daily range > 1.5 * ATR(14)
    wide_range = daily_range > (1.5 * atr_14)
    
    # Weekly trend filter: EMA(21)
    close_1w = df_1w['close'].values
    ema_21 = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Daily volume average
    vol_1d = df_1d['volume'].values
    vol_ma = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any critical data is NaN
        if np.isnan(atr_14[i]) or np.isnan(ema_21[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Get daily index for current daily bar
        idx_1d = i  # Since we're on 1d timeframe, indices match
        
        # Get weekly index for current daily bar (approx: 1d = 1/7 * 1w)
        idx_1w = i // 7
        if idx_1w < 21:  # Need sufficient weekly data for EMA
            continue
        
        # Previous values to avoid look-ahead (use completed daily bar)
        wide_range_prev = wide_range[idx_1d-1]
        ema_prev = ema_21[idx_1w-1]
        vol_ma_prev = vol_ma[idx_1d-1]
        
        if np.isnan(wide_range_prev) or np.isnan(ema_prev) or np.isnan(vol_ma_prev):
            continue
        
        # Create arrays for alignment (constant values for the period)
        wide_range_arr = np.full(len(df_1d), wide_range_prev)
        ema_arr = np.full(len(df_1w), ema_prev)
        vol_ma_arr = np.full(len(df_1d), vol_ma_prev)
        
        # Align to daily timeframe
        wide_range_daily = align_htf_to_ltf(prices, df_1d, wide_range_arr)[i]
        ema_daily = align_htf_to_ltf(prices, df_1w, ema_arr)[i]
        vol_ma_daily = align_htf_to_ltf(prices, df_1d, vol_ma_arr)[i]
        
        if position == 0:
            # Long: wide range breakout above high + weekly uptrend + volume confirmation
            if (high[i] > high_1d[idx_1d-1] and  # break above prior day high
                wide_range_daily and             # prior day was wide range
                close[i] > ema_daily and         # price above weekly EMA (uptrend)
                volume[i] > vol_ma_daily * 1.5): # volume confirmation
                position = 1
                signals[i] = position_size
            # Short: wide range breakdown below low + weekly downtrend + volume confirmation
            elif (low[i] < low_1d[idx_1d-1] and   # break below prior day low
                  wide_range_daily and            # prior day was wide range
                  close[i] < ema_daily and        # price below weekly EMA (downtrend)
                  volume[i] > vol_ma_daily * 1.5): # volume confirmation
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: price closes below prior day low or weekly EMA
            if close[i] < low_1d[idx_1d-1] or close[i] < ema_daily:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: price closes above prior day high or weekly EMA
            if close[i] > high_1d[idx_1d-1] or close[i] > ema_daily:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_Wide_Range_Breakout"
timeframe = "1d"
leverage = 1.0
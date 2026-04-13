#!/usr/bin/env python3
"""
6h_1d1w_Angle_of_Attack
Hypothesis: Price crosses above/below 60-period EMA on 6h chart with 1d volume confirmation and 1w trend filter.
Long when 6h close > 60-period EMA + 1d volume > 1.5x 20-period average + 1w close > 1w EMA50.
Short when 6h close < 60-period EMA + 1d volume > 1.5x 20-period average + 1w close < 1w EMA50.
Exit when price crosses back below/above 60-period EMA or 1w trend reverses.
Designed for 6h timeframe to target 20-40 trades/year with trend-following in both bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # 60-period EMA on 6h
    close_series = pd.Series(close)
    ema_60 = close_series.ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # 1d volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    
    # 1w trend filter using EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(60, n):
        # Skip if any required data is not ready
        if (np.isnan(ema_60[i]) or np.isnan(vol_ma_20_aligned[i]) or np.isnan(vol_1d_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 1d volume > 1.5x 20-period average
        vol_condition = vol_1d_aligned[i] > (vol_ma_20_aligned[i] * 1.5)
        
        # 1w trend condition
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # EMA cross conditions
        ema_cross_up = close[i] > ema_60[i]
        ema_cross_down = close[i] < ema_60[i]
        
        # Exit conditions
        ema_cross_down_exit = close[i] < ema_60[i]  # for long exit
        ema_cross_up_exit = close[i] > ema_60[i]    # for short exit
        trend_reverse_long = close[i] < ema_50_1w_aligned[i]  # uptrend broken
        trend_reverse_short = close[i] > ema_50_1w_aligned[i]  # downtrend broken
        
        if position == 0:
            if ema_cross_up and vol_condition and uptrend:
                position = 1
                signals[i] = position_size
            elif ema_cross_down and vol_condition and downtrend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            if ema_cross_down_exit or trend_reverse_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            if ema_cross_up_exit or trend_reverse_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d1w_Angle_of_Attack"
timeframe = "6h"
leverage = 1.0
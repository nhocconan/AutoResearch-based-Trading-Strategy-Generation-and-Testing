#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme with 1d EMA trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for EMA trend direction.
- Williams %R(14) identifies overbought/oversold conditions: > -20 = overbought, < -80 = oversold.
- In uptrend (price > 1d EMA50): look for short entries when Williams %R > -20 (overbought in trend).
- In downtrend (price < 1d EMA50): look for long entries when Williams %R < -80 (oversold in trend).
- Volume confirmation: current volume > 1.5 * 20-period volume MA to avoid false signals.
- Exit: Opposite Williams %R extreme or trend reversal (price crosses 1d EMA50).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in both bull and bear markets by trading counter-trend extremes within the primary trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA and Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams %R(14) on 1d
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low + 1e-10)
    
    # Align 1d indicators to 6h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 30, 20)  # Need EMA50, Williams %R, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_50_val = ema_50_aligned[i]
        williams_r_val = williams_r_aligned[i]
        curr_close = close[i]
        prev_close = close[i-1]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                # Uptrend: price > 1d EMA50 -> look for short at overbought extreme
                if curr_close > ema_50_val and williams_r_val > -20:
                    signals[i] = -0.25
                    position = -1
                # Downtrend: price < 1d EMA50 -> look for long at oversold extreme
                elif curr_close < ema_50_val and williams_r_val < -80:
                    signals[i] = 0.25
                    position = 1
        elif position == 1:
            # Long exit: price crosses above 1d EMA50 (trend reversal to up) OR Williams %R reaches overbought
            if curr_close > ema_50_val or williams_r_val > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses below 1d EMA50 (trend reversal to down) OR Williams %R reaches oversold
            if curr_close < ema_50_val or williams_r_val < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA50Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0
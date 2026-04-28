#!/usr/bin/env python3
"""
12h_PriceAction_Retracement_Trend_Filter
Hypothesis: Price retracement to key moving averages (EMA21/50) on 12h timeframe with trend filter from 1w EMA200 and volume confirmation works in both bull and bear markets. The 1w trend filter ensures we only trade with the major trend, reducing whipsaw in sideways markets. Price retracement to EMA21/50 provides high-probability entries during pullbacks. Target: 20-40 trades/year per symbol to minimize fee drag while capturing meaningful moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Get 1d data for EMA21/EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA21 and EMA50
    close_1d = df_1d['close'].values
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for all indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(ema_21_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 1w EMA200
        uptrend = close[i] > ema_200_1w_aligned[i]
        downtrend = close[i] < ema_200_1w_aligned[i]
        
        # Price retracement to EMA21 or EMA50 (within 1%)
        near_ema21 = abs(close[i] - ema_21_1d_aligned[i]) / ema_21_1d_aligned[i] < 0.01
        near_ema50 = abs(close[i] - ema_50_1d_aligned[i]) / ema_50_1d_aligned[i] < 0.01
        
        # Entry conditions: retracement to EMA in direction of trend with volume
        long_entry = uptrend and (near_ema21 or near_ema50) and volume_confirmed[i]
        short_entry = downtrend and (near_ema21 or near_ema50) and volume_confirmed[i]
        
        # Exit when price moves away from EMA or trend changes
        long_exit = not uptrend or (close[i] > ema_21_1d_aligned[i] * 1.02 and close[i] > ema_50_1d_aligned[i] * 1.02)
        short_exit = not downtrend or (close[i] < ema_21_1d_aligned[i] * 0.98 and close[i] < ema_50_1d_aligned[i] * 0.98)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_PriceAction_Retracement_Trend_Filter"
timeframe = "12h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 6h strategy combining Bollinger Band breakout with 1d EMA(50) trend filter and volume confirmation.
BB breakouts capture volatility expansion phases, EMA50 filters direction, volume ensures conviction.
Designed for 20-30 trades/year to minimize fee drag while capturing momentum bursts.
Works in bull markets (buy upper BB breakout in uptrend) and bear markets (sell lower BB breakout in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2.0) on 6h close
    bb_length = 20
    bb_mult = 2.0
    bb_mid = np.full(n, np.nan)
    bb_std = np.full(n, np.nan)
    bb_upper = np.full(n, np.nan)
    bb_lower = np.full(n, np.nan)
    
    for i in range(bb_length - 1, n):
        bb_mid[i] = np.mean(close[i - bb_length + 1:i + 1])
        bb_std[i] = np.std(close[i - bb_length + 1:i + 1])
        bb_upper[i] = bb_mid[i] + bb_mult * bb_std[i]
        bb_lower[i] = bb_mid[i] - bb_mult * bb_std[i]
    
    # Get 1d data for EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2/51) + (ema_50_1d[i-1] * 49/51)
    
    # Align 1d EMA to 6h timeframe
    ema_50_1d_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_length, 20, 50)  # need BB, volume MA, EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(ema_50_1d_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below 1d EMA50
        trend_up = close[i] > ema_50_1d_6h[i]
        trend_down = close[i] < ema_50_1d_6h[i]
        
        if position == 0:
            # Long entry: close above upper BB with volume and uptrend
            if (close[i] > bb_upper[i] and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: close below lower BB with volume and downtrend
            elif (close[i] < bb_lower[i] and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: close below middle BB or reverse signal
            if close[i] < bb_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above middle BB or reverse signal
            if close[i] > bb_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BollingerBandBreakout_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with volume confirmation and 1w EMA(34) trend filter.
In bull markets: price above 1w EMA(34) acts as dynamic support, buy on breakouts with volume.
In bear markets: price below 1w EMA(34) acts as resistance, sell on breakdowns with volume.
Weekly EMA(34) filters trend direction to avoid counter-trend trades.
Designed for 10-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    ema = np.full(len(close), np.nan)
    if len(close) < period:
        return ema
    ema[period-1] = np.mean(close[:period])
    for i in range(period, len(close)):
        ema[i] = (close[i] * 2 / (period + 1)) + ema[i-1] * (1 - 2 / (period + 1))
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian(20)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Get 1w data for EMA(34)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Donchian upper/lower on 1d
    upper_1d = np.full(len(high_1d), np.nan)
    lower_1d = np.full(len(low_1d), np.nan)
    for i in range(20, len(high_1d)):
        upper_1d[i] = np.max(high_1d[i-20:i])
        lower_1d[i] = np.min(low_1d[i-20:i])
    
    # Calculate EMA(34) on 1w
    ema_34_1w = calculate_ema(close_1w, 34)
    
    # Align to 1d timeframe (our primary)
    upper_1d_aligned = upper_1d  # already 1d
    lower_1d_aligned = lower_1d  # already 1d
    ema_34_1w_1d = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or 
            np.isnan(ema_34_1w_1d[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper, above weekly EMA, volume confirmation
            if close[i] > upper_1d_aligned[i] and close[i] > ema_34_1w_1d[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower, below weekly EMA, volume confirmation
            elif close[i] < lower_1d_aligned[i] and close[i] < ema_34_1w_1d[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly EMA or Donchian lower
            if close[i] < ema_34_1w_1d[i] or close[i] < lower_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly EMA or Donchian upper
            if close[i] > ema_34_1w_1d[i] or close[i] > upper_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0
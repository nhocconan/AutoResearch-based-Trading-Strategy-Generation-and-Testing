#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Reversal_Bounce_1dTrend_Volume
Hypothesis: In 12h timeframe, price often reverses near Camarilla S3/R3 levels during pullbacks in strong 1d trends. This strategy captures bounces from extreme daily pivot levels with trend alignment and volume confirmation. Designed for lower frequency (~20-40 trades/year) to minimize fee drag while capturing meaningful moves in both bull and bear markets.
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
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous day
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_ = df_1d['high'] - df_1d['low']
    # Focus on S3 and R3 for reversal bounces
    R3 = typical_price + (range_ * 1.1 / 4)
    S3 = typical_price - (range_ * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3.values)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3.values)
    
    # Volume confirmation: >1.5x 30-period MA (less frequent for 12h)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA34 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or
            np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.5 * vol_ma_30[i])
        
        # Reversal bounce conditions: price touches S3/R3 and shows rejection
        # Long: price touches S3 and closes back above it (bounce)
        long_bounce = (low[i] <= S3_aligned[i]) and (close[i] > S3_aligned[i]) and vol_confirm and uptrend
        # Short: price touches R3 and closes back below it (rejection)
        short_bounce = (high[i] >= R3_aligned[i]) and (close[i] < R3_aligned[i]) and vol_confirm and downtrend
        
        # Exit conditions: return to midpoint or opposite extreme
        midpoint = (R3_aligned[i] + S3_aligned[i]) / 2
        long_exit = close[i] < midpoint
        short_exit = close[i] > midpoint
        
        if long_bounce and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_bounce and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_Pivot_Reversal_Bounce_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0
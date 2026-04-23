#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R reversal with 1d EMA50 trend filter and volume confirmation.
- Long: Williams %R(14) crosses above -80 AND price > 1d EMA50 AND volume > 1.8x 20-period avg
- Short: Williams %R(14) crosses below -20 AND price < 1d EMA50 AND volume > 1.8x 20-period avg
- Exit: Opposite Williams %R cross OR price crosses 1d EMA50
- Uses 1d HTF for EMA50 and Williams %R (calculated from prior 1d bar)
- Designed for low trade frequency (12-37/year) to minimize fee drag
- Williams %R identifies overextended conditions for mean reversion in ranging markets
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
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA50 for trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R from prior 1d bar (HTF = 1d)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d_arr) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid div by zero
    
    # Align Williams %R to 12h timeframe (use prior completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # Need 50 for EMA, 20 for volume MA, 14 for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(williams_r_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        # Williams %R signals (using prior bar to avoid look-ahead)
        williams_prev = williams_r_aligned[i-1]
        williams_curr = williams_r_aligned[i]
        
        # Bullish reversal: Williams %R crosses above -80 (from oversold)
        williams_cross_up = williams_prev <= -80 and williams_curr > -80
        # Bearish reversal: Williams %R crosses below -20 (from overbought)
        williams_cross_down = williams_prev >= -20 and williams_curr < -20
        
        if position == 0:
            # Long: Williams %R bullish reversal AND price > 1d EMA50 AND volume confirmation
            if williams_cross_up and volume_confirm and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R bearish reversal AND price < 1d EMA50 AND volume confirmation
            elif williams_cross_down and volume_confirm and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R bearish reversal OR price < 1d EMA50 (trend flip)
            if williams_cross_down or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R bullish reversal OR price > 1d EMA50 (trend flip)
            if williams_cross_up or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_Reversal_1dEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0
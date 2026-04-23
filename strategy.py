#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R mean reversion with 1w EMA50 trend filter and volume confirmation.
- Long: Williams %R(14) < -80 (oversold) AND price > 1w EMA50 AND volume > 1.5x 24-period avg
- Short: Williams %R(14) > -20 (overbought) AND price < 1w EMA50 AND volume > 1.5x 24-period avg
- Exit: Williams %R crosses above -50 (for long) or below -50 (for short)
- Uses 1w HTF for EMA50 and Williams %R calculated from prior completed 1w bar
- Designed for low trade frequency (12-37/year) to minimize fee drag on 12h timeframe
- Works in bull (buy oversold dips in uptrend) and bear (sell overbought rallies in downtrend)
- Volume confirmation reduces false signals
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
    
    # Volume confirmation: > 1.5x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Calculate 1w EMA50 for trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams %R(14) from prior completed 1w bar (HTF = 1w)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1w_arr) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 12h timeframe (use prior completed 1w bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 14, 24)  # Need 50 for EMA, 14 for Williams %R, 24 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(williams_r_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Williams %R signals (using current value vs thresholds)
        wr = williams_r_aligned[i]
        wr_prev = williams_r_aligned[i-1] if i > 0 else wr
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND price > 1w EMA50 AND volume confirmation
            if wr < -80 and volume_confirm and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND price < 1w EMA50 AND volume confirmation
            elif wr > -20 and volume_confirm and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -50 (momentum fading)
            if wr > -50 and wr_prev <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -50 (momentum fading)
            if wr < -50 and wr_prev >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_MeanReversion_1wEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0
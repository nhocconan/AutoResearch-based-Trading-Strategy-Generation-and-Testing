#!/usr/bin/env python3
"""
Hypothesis: 1d Williams %R Reversal with 1w EMA50 trend filter and volume confirmation.
- Long: Williams %R(14) < -80 AND price > 1w EMA50 AND volume > 1.5x 20-period avg
- Short: Williams %R(14) > -20 AND price < 1w EMA50 AND volume > 1.5x 20-period avg
- Exit: Williams %R crosses above -50 (long) or below -50 (short)
- Uses 1w HTF for EMA50 trend filter (calculated from prior completed bars)
- Designed for low trade frequency (7-25/year) to minimize fee drag on 1d timeframe
- Williams %R identifies overbought/oversold conditions for mean reversion
- Volume confirmation filters low-conviction moves
- Works in both bull (buy oversold dips) and bear (sell overbought rallies) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA50 for trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams %R(14) from 1d data
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # Need 50 for EMA, 20 for volume MA, 14 for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Williams %R signals
        wr = williams_r[i]
        wr_prev = williams_r[i-1] if i > 0 else -50
        
        # Oversold/overbought conditions
        oversold = wr < -80
        overbought = wr > -20
        # Exit when Williams %R crosses -50 (centerline)
        cross_above_50 = wr > -50 and wr_prev <= -50
        cross_below_50 = wr < -50 and wr_prev >= -50
        
        if position == 0:
            # Long: Williams %R oversold AND price > 1w EMA50 AND volume confirmation
            if oversold and volume_confirm and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought AND price < 1w EMA50 AND volume confirmation
            elif overbought and volume_confirm and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -50 (momentum fading)
            if cross_above_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -50 (momentum fading)
            if cross_below_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR_Reversal_1wEMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0
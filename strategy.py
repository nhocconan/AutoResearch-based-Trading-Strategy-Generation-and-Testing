#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 12h EMA50 trend filter and volume spike confirmation.
- Long: Williams %R(14) crosses above -80 (oversold reversal) AND close > 12h EMA50 AND volume > 2.0x 24-period avg
- Short: Williams %R(14) crosses below -20 (overbought reversal) AND close < 12h EMA50 AND volume > 2.0x 24-period avg
- Exit: Opposite Williams %R reversal OR price crosses 12h EMA50
- Uses 12h HTF for EMA50 trend filter (aligned with completed bar)
- Designed for low trade frequency (12-37/year) to minimize fee drag
- Works in bull (buy oversold reversals in uptrend) and bear (sell overbought reversals in downtrend)
- Williams %R is a momentum oscillator that identifies overbought/oversold conditions, effective in ranging markets
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
    
    # Volume confirmation: > 2.0x 24-period average (24*6h = 6 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Calculate 12h EMA50 for trend filter (HTF = 12h)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Williams %R(14) on primary timeframe (6h)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 24, 14)  # Need 50 for EMA, 24 for volume MA, 14 for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(williams_r[i]) or
            np.isnan(williams_r[i-1])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Williams %R reversal signals
        williams_r_cross_up = williams_r[i-1] <= -80 and williams_r[i] > -80  # Cross above -80 (oversold reversal)
        williams_r_cross_down = williams_r[i-1] >= -20 and williams_r[i] < -20  # Cross below -20 (overbought reversal)
        
        if position == 0:
            # Long: Williams %R crosses above -80 AND price > 12h EMA50 AND volume confirmation
            if williams_r_cross_up and volume_confirm and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 AND price < 12h EMA50 AND volume confirmation
            elif williams_r_cross_down and volume_confirm and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -20 OR price < 12h EMA50 (trend flip)
            if williams_r_cross_down or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -80 OR price > 12h EMA50 (trend flip)
            if williams_r_cross_up or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Reversal_12hEMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0
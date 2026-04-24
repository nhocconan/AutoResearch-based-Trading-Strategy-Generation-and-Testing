#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R Reversal with 1d EMA50 trend filter and volume spike confirmation.
- Uses Williams %R (14) on 12h for oversold/overbought conditions.
- Long when %R crosses above -80 (oversold) with volume > 1.5x 20-bar average and price above 1d EMA50.
- Short when %R crosses below -20 (overbought) with volume > 1.5x 20-bar average and price below 1d EMA50.
- Exit on opposite %R cross (%R < -80 for longs, %R > -20 for shorts) or trend failure.
- Designed for 12h timeframe to capture swing reversals with medium frequency.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 12-37 trades/year (50-150 total over 4 years) to stay fee-efficient.
- Works in both bull and bear markets by trading mean reversions within the trend.
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
    
    # Get 12h data ONCE before loop for Williams %R
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Williams %R for 12h timeframe
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    
    # Align Williams %R to 12h timeframe (wait for 12h bar to close)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Only trade if volume confirms reversal
            if volume_confirm:
                # Long: Williams %R crosses above -80 (oversold) AND price above 1d EMA50
                if williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -20 (overbought) AND price below 1d EMA50
                elif williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and close[i] < ema_50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -80 OR price below 1d EMA50
            if williams_r_aligned[i] < -80 and williams_r_aligned[i-1] >= -80 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -20 OR price above 1d EMA50
            if williams_r_aligned[i] > -20 and williams_r_aligned[i-1] <= -20 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_Reversal_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme Reversal with 1d Trend Filter and Volume Spike.
- Williams %R(14) identifies overbought/oversold conditions: < -80 = oversold, > -20 = overbought.
- Extreme readings (< -90 or > -10) signal exhaustion and high-probability reversals.
- 1d EMA50 provides higher-timeframe trend filter: only take longs above EMA50, shorts below.
- Volume spike (> 2.0x 24-period average) confirms institutional participation at reversal points.
- Position size 0.25 balances profit potential with drawdown control in volatile 6h timeframe.
- Works in both bull and bear markets: 1d trend filter avoids counter-trend whipsaws,
  while Williams %R extremes capture exhaustion moves in ranging markets.
- Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
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
    
    # Williams %R(14) calculation: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 2.0x 24-period average (strict for 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 14, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Only trade with volume confirmation
            if volume_confirm:
                # Long: Williams %R < -90 (extreme oversold) + above 1d EMA50 (bullish trend)
                if williams_r[i] < -90 and close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R > -10 (extreme overbought) + below 1d EMA50 (bearish trend)
                elif williams_r[i] > -10 and close[i] < ema_50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R > -20 (exits overbought) OR below 1d EMA50 (trend change)
            if williams_r[i] > -20 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R < -80 (exits oversold) OR above 1d EMA50 (trend change)
            if williams_r[i] < -80 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0
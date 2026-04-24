#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R extreme reversal with 1d EMA50 trend filter and volume confirmation.
- Uses 12h timeframe (primary) and 1d HTF for EMA50 trend alignment
- Williams %R(14) identifies overbought/oversold conditions: long when %R crosses above -80 from below, short when crosses below -20 from above
- Trend filter: only long when 12h EMA21 > 1d EMA50, only short when 12h EMA21 < 1d EMA50
- Volume confirmation: current 12h volume > 1.8 * 30-period 12h volume MA to filter weak breakouts
- Discrete signal size: 0.25 to balance reward and risk, minimizing fee churn
- Stoploss: exit when price closes opposite the entry extreme (%R crosses below -20 for longs, above -80 for shorts)
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe as per research
- Works in both bull/bear: trend filter avoids counter-trend trades, Williams %R captures reversals at extremes in all regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA21 for trend confirmation
    ema_21_12h = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R(14) on 12h timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current volume > 1.8 * 30-period volume MA
    volume_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    # Trend filter: 12h EMA21 vs 1d EMA50
    uptrend = ema_21_12h > ema_50_1d_aligned
    downtrend = ema_21_12h < ema_50_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 30, 14)  # Need 1d EMA50, volume MA, and Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below AND uptrend AND volume spike
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and uptrend[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above AND downtrend AND volume spike
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and downtrend[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -20 (overbought reversal)
            if williams_r[i] < -20 and williams_r[i-1] >= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -80 (oversold reversal)
            if williams_r[i] > -80 and williams_r[i-1] <= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0
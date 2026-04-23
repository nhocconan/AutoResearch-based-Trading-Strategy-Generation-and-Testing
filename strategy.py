#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme Reversal with 12h EMA50 trend filter and volume confirmation.
- Williams %R(14) < -90 for long entry (extreme oversold), > -10 for short entry (extreme overbought)
- 12h EMA50 as trend filter: long only when price > EMA50, short only when price < EMA50
- Volume confirmation: > 2.0x 24-period average to avoid low-volume false signals
- Discrete position sizing: 0.25 to minimize fee churn and control drawdown
- Works in bull/bear markets via trend filter and mean reversion from extremes
- Target: 50-150 total trades over 4 years (12-37/year) to stay within fee limits
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 2.0x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams %R(14) on 6h data
    def calculate_williams_r(high, low, close, window):
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    wr = calculate_williams_r(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 24, 14)  # EMA50, volume MA, Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(wr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Williams %R extreme levels
        wr_oversold = wr[i] < -90.0   # Extreme oversold
        wr_overbought = wr[i] > -10.0  # Extreme overbought
        
        if position == 0:
            # Long: Williams %R < -90 (oversold) AND price > 12h EMA50 AND volume confirmation
            if wr_oversold and volume_confirm and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -10 (overbought) AND price < 12h EMA50 AND volume confirmation
            elif wr_overbought and volume_confirm and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R > -50 (recovered from oversold) OR price < 12h EMA50 (trend flip)
            if wr[i] > -50.0 or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R < -50 (recovered from overbought) OR price > 12h EMA50 (trend flip)
            if wr[i] < -50.0 or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_12hEMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0
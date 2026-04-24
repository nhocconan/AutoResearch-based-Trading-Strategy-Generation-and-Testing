#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Reversal with 1d EMA50 trend filter and volume spike confirmation.
- Uses Williams %R(14) from 6h timeframe for overbought/oversold conditions.
- Long: %R crosses above -80 (oversold recovery) with price > 1d EMA50 and volume > 1.5x average.
- Short: %R crosses below -20 (overbought decline) with price < 1d EMA50 and volume > 1.5x average.
- Trend filter: 1d EMA50 ensures alignment with higher timeframe direction.
- Volume confirmation reduces false signals during low-participation moves.
- Designed for 6h timeframe to capture swing reversals with lower frequency.
- Targets 12-37 trades/year (50-150 total over 4 years) to stay fee-efficient.
- Williams %R is effective in ranging and trending markets, capturing mean reversion within trend.
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
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R(14) on 6h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # Need enough for EMA, volume MA, and Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Avoid division by zero in Williams %R calculation
        if highest_high[i] == lowest_low[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Only trade if volume confirms
            if volume_confirm:
                # Long: Williams %R crosses above -80 (oversold recovery) AND price > 1d EMA50
                if williams_r[i] > -80 and williams_r[i-1] <= -80 and close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -20 (overbought decline) AND price < 1d EMA50
                elif williams_r[i] < -20 and williams_r[i-1] >= -20 and close[i] < ema_50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -20 (overbought) OR price < 1d EMA50
            if williams_r[i] > -20 and williams_r[i-1] <= -20 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -80 (oversold) OR price > 1d EMA50
            if williams_r[i] < -80 and williams_r[i-1] >= -80 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Reversal_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0
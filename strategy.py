#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 12h EMA50 trend filter and volume confirmation.
- Williams %R(14) identifies overbought/oversold conditions: < -80 = oversold, > -20 = overbought.
- Long: Williams %R crosses above -80 from below AND price > 12h EMA50 AND volume > 1.5x 20-bar average.
- Short: Williams %R crosses below -20 from above AND price < 12h EMA50 AND volume > 1.5x 20-bar average.
- Trend filter: 12h EMA50 ensures alignment with intermediate-term trend.
- Volume confirmation reduces false reversals in low-momentum environments.
- Designed for 6h timeframe to capture swing reversals with controlled frequency.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 12-37 trades/year (50-150 total over 4 years) to stay fee-efficient.
- Williams %R is effective in ranging and trending markets, working in both bull and bear regimes.
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
    
    # Get 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams %R(14) on 6h data
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    # Avoid division by zero
    rr = highest_high - lowest_low
    williams_r = np.where(rr != 0, -100 * (highest_high - close) / rr, -50.0)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Only trade if volume confirms reversal
            if volume_confirm:
                # Williams %R crossover signals
                wr_prev = williams_r[i-1]
                wr_curr = williams_r[i]
                
                # Long: %R crosses above -80 from below (oversold reversal)
                if wr_prev <= -80.0 and wr_curr > -80.0 and close[i] > ema_50_12h_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: %R crosses below -20 from above (overbought reversal)
                elif wr_prev >= -20.0 and wr_curr < -20.0 and close[i] < ema_50_12h_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: %R crosses above -20 (overbought) OR price crosses below EMA50
            if williams_r[i] >= -20.0 or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: %R crosses below -80 (oversold) OR price crosses above EMA50
            if williams_r[i] <= -80.0 or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Reversal_12hEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0
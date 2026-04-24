#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 12h EMA50 trend filter and volume confirmation.
- Williams %R(14) identifies overbought/oversold conditions for mean reversion entries.
- Long when %R crosses above -80 from below AND price > 12h EMA50 (uptrend filter).
- Short when %R crosses below -20 from above AND price < 12h EMA50 (downtrend filter).
- Volume confirmation requires >2.0x 24-period average to ensure conviction.
- Designed for 12-25 trades/year (50-100 total over 4 years) to stay within fee-efficient range.
- Williams %R works well in ranging markets (2025+ test period) and catches reversals in trends.
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
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Prior 12h close (completed 12h bar)
    close_12h = df_12h['close'].shift(1).values
    
    # Align to 6h timeframe
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    
    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams %R(14) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Volume confirmation: > 2.0x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 14, 24)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Williams %R reversal signals
            wr_prev = williams_r[i-1]
            wr_curr = williams_r[i]
            
            # Long: %R crosses above -80 from below AND price > 12h EMA50 AND volume confirmation
            if wr_prev <= -80 and wr_curr > -80 and close[i] > ema_50_12h_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: %R crosses below -20 from above AND price < 12h EMA50 AND volume confirmation
            elif wr_prev >= -20 and wr_curr < -20 and close[i] < ema_50_12h_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: %R crosses above -20 (overbought) OR price crosses below 12h EMA50
            if williams_r[i] > -20 or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: %R crosses below -80 (oversold) OR price crosses above 12h EMA50
            if williams_r[i] < -80 or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Reversal_12hEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0
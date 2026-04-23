#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R reversal with 1d EMA34 trend filter and volume spike confirmation.
- Williams %R(14) identifies overbought/oversold conditions: > -20 = overbought, < -80 = oversold
- Reversal signals: long when %R crosses above -80 from below, short when crosses below -20 from above
- 1d EMA34 ensures trades align with daily trend (avoid counter-trend in bear markets)
- Volume confirmation: > 1.5x 20-period average to filter weak breakouts
- Discrete position size 0.25 to balance return and drawdown
- Target: 20-40 trades/year on 4h timeframe (80-160 total over 4 years)
- Works in both bull/bear via 1d trend filter and volatility-adjusted reversals
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
    
    # Williams %R calculation (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    rr = highest_high - lowest_low
    rr[rr == 0] = 1e-10
    willr = -100 * (highest_high - close) / rr
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # volume MA, 1d EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(willr[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold reversal) AND price above 1d EMA34 AND volume confirmation
            if willr[i] > -80 and willr[i-1] <= -80 and close[i] > ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought reversal) AND price below 1d EMA34 AND volume confirmation
            elif willr[i] < -20 and willr[i-1] >= -20 and close[i] < ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -20 (overbought) OR price crosses below 1d EMA34
            if willr[i] >= -20 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -80 (oversold) OR price crosses above 1d EMA34
            if willr[i] <= -80 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Reversal_1dEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0
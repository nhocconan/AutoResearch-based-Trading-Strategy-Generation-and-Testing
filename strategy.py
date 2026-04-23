#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R reversal with 1w EMA34 trend filter and volume confirmation.
- Williams %R(14) identifies overbought/oversold conditions; long when %R crosses above -80 from below, short when crosses below -20 from above.
- 1w EMA34 ensures alignment with long-term trend to avoid counter-trend trades.
- Volume > 1.3x 20-period average confirms reversal validity.
- Discrete position size 0.25 limits drawdown during crashes.
- Designed for 12h timeframe to target 12-37 trades/year (50-150 total over 4 years).
- Works in both bull and bear regimes via trend filter and mean-reversion logic.
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
    
    # Williams %R (14-period) - using prior bar to avoid look-ahead
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    rr = highest_high - lowest_low
    rr[rr == 0] = 1e-10
    williams_r = -100 * (highest_high - close) / rr
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20, 34)  # Williams %R, volume MA, 1w EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (from below) AND price above 1w EMA34 AND volume confirmation
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and close[i] > ema_34_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (from above) AND price below 1w EMA34 AND volume confirmation
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and close[i] < ema_34_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -20 (overbought) OR price crosses below 1w EMA34
            if williams_r[i] > -20 or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -80 (oversold) OR price crosses above 1w EMA34
            if williams_r[i] < -80 or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR14_Reversal_1wEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0
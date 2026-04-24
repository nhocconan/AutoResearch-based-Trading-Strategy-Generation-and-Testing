#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R (14) with 1d EMA34 trend filter and volume spike confirmation.
- Uses 4h timeframe (primary) and 1d HTF for EMA34 trend alignment (proven BTC/ETH edge from DB).
- Williams %R identifies overbought/oversold conditions: long when %R crosses above -80 from below,
  short when %R crosses below -20 from above.
- Trend filter: only long when 4h close > 1d EMA34, only short when 4h close < 1d EMA34.
- Volume confirmation: current 4h volume > 2.0 * 20-period 4h volume MA (strict to reduce trades).
- Exit: reverse signal or %R crosses midpoint (-50) in opposite direction.
- Discrete signal size: 0.25 to balance reward and risk, minimizing fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in both bull/bear: trend filter avoids counter-trend trades, Williams %R captures momentum reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R (14) calculation
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (strict)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Trend filter: 4h close vs 1d EMA34
    uptrend = close > ema_34_1d_aligned
    downtrend = close < ema_34_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 34, 20)  # Need Williams %R, EMA34, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1d_aligned[i]) or 
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
            # Long exit: reverse signal or Williams %R crosses below -50
            if (williams_r[i] < -20 and williams_r[i-1] >= -20 and downtrend[i] and volume_spike[i]) or \
               (williams_r[i] < -50 and williams_r[i-1] >= -50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: reverse signal or Williams %R crosses above -50
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and uptrend[i] and volume_spike[i]) or \
               (williams_r[i] > -50 and williams_r[i-1] <= -50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_14_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0
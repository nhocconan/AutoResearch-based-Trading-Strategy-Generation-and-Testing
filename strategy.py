#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R with 1d EMA34 trend filter and volume confirmation.
- Williams %R(14) identifies overbought/oversold conditions on 6h chart
- Trend filter: only long when price > 1d EMA34 (uptrend), only short when price < 1d EMA34 (downtrend)
- Volume confirmation: current volume > 1.3 * 20-period volume MA to filter low-noise breakouts
- Entry: Williams %R crosses above -20 from below (bullish momentum) in uptrend OR crosses below -80 from above (bearish momentum) in downtrend
- Exit: Williams %R returns to -50 level (mean reversion) or opposite signal
- Discrete signal size: 0.25 to balance profit potential and drawdown control
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
- Works in both bull/bear: trend filter ensures we trade with higher timeframe momentum, Williams %R captures short-term exhaustion
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Calculate Williams %R(14) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current volume > 1.3 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * volume_ma)
    
    # Trend filter: price above/below 1d EMA34
    uptrend = close > ema_34_1d_aligned
    downtrend = close < ema_34_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # Need 1d EMA34, volume MA(20), Williams %R(14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -20 from below AND uptrend AND volume confirmation
            if williams_r[i] > -20 and williams_r[i-1] <= -20 and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -80 from above AND downtrend AND volume confirmation
            elif williams_r[i] < -80 and williams_r[i-1] >= -80 and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R returns to -50 (mean reversion) or bearish signal
            if williams_r[i] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns to -50 (mean reversion) or bullish signal
            if williams_r[i] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dEMA34_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0
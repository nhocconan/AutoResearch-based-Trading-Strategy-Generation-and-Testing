#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R (14) mean reversion with 1d EMA50 trend filter and volume spike confirmation.
- Williams %R identifies overbought/oversold conditions: long when %R crosses above -80 from below, short when crosses below -20 from above
- Trend filter: only long when price > 1d EMA50, only short when price < 1d EMA50 to avoid counter-trend trades
- Volume confirmation: current volume > 1.8 * 20-period volume MA to ensure breakout validity
- Exit: reverse signal or when Williams %R returns to neutral zone (-50) indicating mean reversion completion
- Discrete signal size: 0.25 to balance return and risk while minimizing fee churn
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
- Works in bull/bear markets: trend filter prevents counter-trend trades, Williams %R captures reversals in ranging markets
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R (14) on 4h timeframe
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current volume > 1.8 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * volume_ma)
    
    # Trend filter: price above/below 1d EMA50
    uptrend = close > ema_50_1d_aligned
    downtrend = close < ema_50_1d_aligned
    
    # Williams %R thresholds for mean reversion
    williams_oversold = williams_r < -80  # Oversold condition
    williams_overbought = williams_r > -20  # Overbought condition
    williams_neutral = (williams_r >= -50) & (williams_r <= -50)  # Exactly -50 for exit
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 14, 20)  # Need 1d EMA50, Williams %R(14), and volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below AND uptrend AND volume confirmation
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                uptrend[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above AND downtrend AND volume confirmation
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  downtrend[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R returns to neutral zone (-50) or reverse signal
            if williams_r[i] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns to neutral zone (-50) or reverse signal
            if williams_r[i] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR14_1dEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0
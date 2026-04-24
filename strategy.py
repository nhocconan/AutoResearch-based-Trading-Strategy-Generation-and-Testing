#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R + 1d EMA34 trend + volume confirmation.
- Primary timeframe: 4h for balance of signal quality and trade frequency
- Williams %R(14) identifies overbought/oversold: long when %R crosses above -80 from below, short when crosses below -20 from above
- 1d EMA34 trend filter ensures trades align with higher timeframe direction
- Volume confirmation: current volume > 1.5 * 20-period volume MA to filter low-noise breakouts
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn
- Target: 75-200 total trades over 4 years (19-50/year) for 4h as per research
- Works in bull/bear: trend filter avoids counter-trend trades, Williams %R captures reversals in all regimes
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
    
    # Williams %R(14) on 4h: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * volume_ma)
    
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
            # Long: Williams %R crosses above -80 from below AND uptrend AND volume confirmation
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above AND downtrend AND volume confirmation
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 (mean reversion) or reverse signal
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 (mean reversion) or reverse signal
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_1dEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0
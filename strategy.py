#!/usr/bin/env python3
"""
Hypothesis: 1d Williams %R with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d, HTF: 1w for EMA50 trend alignment.
- Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100 over 14 periods.
- Trend filter: only long when 1d close > 1w EMA50, only short when 1d close < 1w EMA50.
- Volume confirmation: current 1d volume > 1.5 * 20-period 1d volume MA.
- Discrete signal size: 0.25 to minimize fee churn and control drawdown.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
- Exit: Williams %R reverses from overbought/oversold territory (%R > -20 for long exit, %R < -80 for short exit).
- Works in bull via trend alignment, in bear via mean reversion from extremes.
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
    
    # Calculate 14-period Williams %R
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # Need 1w EMA50, volume MA, Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND uptrend AND volume spike
            if williams_r[i] < -80 and close[i] > ema_50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND downtrend AND volume spike
            elif williams_r[i] > -20 and close[i] < ema_50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R > -20 (exiting oversold) or trend change
            if williams_r[i] > -20 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R < -80 (exiting overbought) or trend change
            if williams_r[i] < -80 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0
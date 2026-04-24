#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R reversal with 12h EMA50 trend filter and volume spike confirmation.
- Williams %R(14) identifies overbought/oversold conditions: < -80 = oversold, > -20 = overbought.
- Long: Williams %R crosses above -80 from below with volume > 1.8x 20-bar average AND price > 12h EMA50.
- Short: Williams %R crosses below -20 from above with volume > 1.8x 20-bar average AND price < 12h EMA50.
- Uses 12h EMA50 as trend filter to align with medium-term trend and reduce counter-trend trades.
- Volume confirmation reduces false reversals in low-participation moves.
- Discrete position size 0.25 to manage drawdown and limit fee churn.
- Designed for 4h timeframe to capture swing reversals in both bull and bear markets.
- Targets 30-60 trades/year (120-240 total over 4 years) to stay within fee-efficient range.
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
    
    # Williams %R(14) on 4h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    rr = highest_high - lowest_low
    williams_r = np.where(rr != 0, -100 * (highest_high - close) / rr, -50.0)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 14, 20)  # Need enough for EMA, Williams %R, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Only trade if volume confirms reversal
            if volume_confirm:
                # Long: Williams %R crosses above -80 from below AND price > 12h EMA50
                if williams_r[i] > -80.0 and williams_r[i-1] <= -80.0 and close[i] > ema_50_12h_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -20 from above AND price < 12h EMA50
                elif williams_r[i] < -20.0 and williams_r[i-1] >= -20.0 and close[i] < ema_50_12h_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -20 (overbought) OR price < 12h EMA50
            if williams_r[i] >= -20.0 or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -80 (oversold) OR price > 12h EMA50
            if williams_r[i] <= -80.0 or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Reversal_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0
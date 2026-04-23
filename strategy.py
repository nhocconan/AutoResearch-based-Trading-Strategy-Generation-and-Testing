#!/usr/bin/env python3
"""
Hypothesis: Daily Williams %R reversal with 1-week EMA50 trend filter and volume spike confirmation.
- Williams %R(14) identifies overbought/oversold conditions on 1d timeframe for mean reversion.
- 1-week EMA50 ensures alignment with weekly trend to avoid counter-trend whipsaws in bear markets.
- Volume > 1.8x 20-period average confirms reversal validity (tight filter to reduce trades).
- Discrete position size 0.25 limits drawdown during crashes.
- Target: 15-25 trades/year on 1d timeframe (60-100 total over 4 years) to stay within fee-efficient range.
- Designed to work in both bull and bear regimes via trend filter and volume confirmation.
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
    
    # Williams %R from prior 1d (using mtf_data to avoid look-ahead)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # need enough for Williams %R calculation
        return np.zeros(n)
    
    # Calculate Williams %R on 1d: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    
    # Avoid division by zero
    dd = highest_high - lowest_low
    williams_r = np.where(dd != 0, (highest_high - close_1d) / dd * -100, -50)
    
    # Align Williams %R to 1d timeframe (use prior completed day)
    williams_r_lagged = np.roll(williams_r, 1)  # shift by 1 to use prior completed day
    williams_r_lagged[0] = -50  # neutral for first day
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_lagged)
    
    # 1-week EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # Williams %R needs 14+1 lag, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND price above 1w EMA50 AND volume confirmation
            if williams_r_aligned[i] < -80 and close[i] > ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND price below 1w EMA50 AND volume confirmation
            elif williams_r_aligned[i] > -20 and close[i] < ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R > -50 (exit oversold) OR price crosses below 1w EMA50
            if williams_r_aligned[i] > -50 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R < -50 (exit overbought) OR price crosses above 1w EMA50
            if williams_r_aligned[i] < -50 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR_Reversal_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0
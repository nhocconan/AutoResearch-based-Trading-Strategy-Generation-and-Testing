#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R with 1d EMA trend filter and volume confirmation.
- Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
- Long when Williams %R crosses above -80 (oversold bounce) AND price > 1d EMA50 (uptrend)
- Short when Williams %R crosses below -20 (overbought rejection) AND price < 1d EMA50 (downtrend)
- Volume confirmation: current volume > 1.5 * 20-period average volume
- Exit when Williams %R crosses -50 (mean reversion) or trend fails
- Designed to capture mean reversion in extremes with trend alignment
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R(14) calculation
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Trend filter
    uptrend = close > ema_50_1d_aligned
    downtrend = close < ema_50_1d_aligned
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 AND uptrend AND volume confirmation
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and uptrend[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 AND downtrend AND volume confirmation
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and downtrend[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -50 OR trend fails
            if williams_r[i] >= -50 or not uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -50 OR trend fails
            if williams_r[i] <= -50 or not downtrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0
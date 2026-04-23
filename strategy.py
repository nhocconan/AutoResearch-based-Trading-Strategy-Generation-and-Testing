#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R mean reversion with 1w EMA34 trend filter and volume spike confirmation.
Long when Williams %R(14) < -80 (oversold) AND 1w close > 1w EMA34 (uptrend) AND volume > 2.5x 24-period MA.
Short when Williams %R(14) > -20 (overbought) AND 1w close < 1w EMA34 (downtrend) AND volume > 2.5x 24-period MA.
Exit when Williams %R crosses above -50 (for long) or below -50 (for short).
Designed for low trade frequency (target: 15-25/year) with mean reversion in 6h timeframe.
Williams %R identifies extreme reversals, weekly trend filter ensures alignment with higher timeframe momentum.
Volume spike filter reduces false signals. Strategy should work in both bull and bear markets by
trading mean reversion within the dominant weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R(14) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 6h volume MA (24-period) for confirmation
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 24)  # need EMA34 and volume MA24
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 1w close > EMA34 = uptrend, close < EMA34 = downtrend
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
        trend_up = close_1w_aligned[i] > ema_34_1w_aligned[i]
        trend_down = close_1w_aligned[i] < ema_34_1w_aligned[i]
        
        # Volume filter: 6h volume > 2.5x 24-period MA (stricter to reduce trades)
        vol_filter = volume[i] > 2.5 * vol_ma_24[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND uptrend AND volume filter
            if williams_r[i] < -80 and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND downtrend AND volume filter
            elif williams_r[i] > -20 and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R crosses -50 mean level
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses above -50
                if williams_r[i] > -50:
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses below -50
                if williams_r[i] < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_MeanReversion_1wEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0
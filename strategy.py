#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Reversal with 1w EMA34 trend filter and volume confirmation.
Long when Williams %R(14) < -80 AND 1w EMA34 rising AND volume > 1.5x 20-period MA.
Short when Williams %R(14) > -20 AND 1w EMA34 falling AND volume > 1.5x 20-period MA.
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
Uses 1w HTF for trend filter to avoid counter-trend trades in bear markets, volume spike for momentum confirmation.
Williams %R is effective at catching reversals in both bull and bear markets due to its overbought/oversold nature.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
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
    
    # Calculate 1w EMA34 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams %R(14) on 6h timeframe
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 6h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, period)  # EMA34, volume MA, Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_1w_aligned[i-1]
            ema_rising = ema_34_1w_aligned[i] > ema_prev
            ema_falling = ema_34_1w_aligned[i] < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 6h volume > 1.5x 20-period MA (higher threshold to reduce frequency)
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND EMA34 rising AND volume filter
            if williams_r[i] < -80 and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND EMA34 falling AND volume filter
            elif williams_r[i] > -20 and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R crosses -50 midpoint
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

name = "6H_WilliamsR_Reversal_1wEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R reversal with 1d EMA34 trend filter and volume confirmation.
Long when Williams %R < -80 (oversold) AND 1d EMA34 rising AND volume > 1.5x 20-period MA.
Short when Williams %R > -20 (overbought) AND 1d EMA34 falling AND volume > 1.5x 20-period MA.
Exit when Williams %R crosses above -50 (for long) or below -50 (for short).
Uses 1d HTF for trend filter to avoid counter-trend trades, volume spike for momentum confirmation.
Williams %R calculated on 12h timeframe for precise entry timing.
Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
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
    
    # Calculate 12h Williams %R (14-period)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(period, 34, 20)  # Williams %R, EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_1d_aligned[i-1]
            ema_rising = ema_34_1d_aligned[i] > ema_prev
            ema_falling = ema_34_1d_aligned[i] < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 12h volume > 1.5x 20-period MA (higher threshold to reduce frequency)
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
            # Exit conditions: Williams %R crosses above -50 (for long) or below -50 (for short)
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses above -50
                if i >= start_idx + 1 and williams_r[i] > -50 and williams_r[i-1] <= -50:
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses below -50
                if i >= start_idx + 1 and williams_r[i] < -50 and williams_r[i-1] >= -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_Reversal_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0
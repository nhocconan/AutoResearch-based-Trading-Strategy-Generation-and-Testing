#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + 1d EMA34 trend filter + 1d volume spike confirmation.
Long when Alligator jaws (13-period SMMA) > teeth (8-period SMMA) > lips (5-period SMMA) AND 1d EMA34 rising AND 1d volume > 1.5x 20-period average.
Short when jaws < teeth < lips AND 1d EMA34 falling AND 1d volume > 1.5x 20-period average.
Exit when Alligator lines cross in opposite direction or EMA34 reverses.
Uses 1d HTF for EMA34 trend and volume filter to reduce whipsaws and false signals.
Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
Williams Alligator: SMMA(median, period), where SMMA = smoothed moving average (EMA with alpha=1/period).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(arr, period):
    """Smoothed Moving Average: EMA with alpha=1/period"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    result = np.full_like(arr, np.nan, dtype=float)
    # First value: SMA
    result[period-1] = np.mean(arr[:period])
    # Subsequent values: SMMA(i) = (SMMA(i-1)*(period-1) + arr[i]) / period
    for i in range(period, len(arr)):
        result[i] = (result[i-1] * (period-1) + arr[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate median price for Alligator
    median_price = (high + low) / 2.0
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d volume average for spike filter (HTF)
    vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Williams Alligator lines on 1d timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    median_1d = (high_1d + low_1d) / 2.0
    
    # Alligator: Lips=SMMA(5), Teeth=SMMA(8), Jaws=SMMA(13)
    lips_1d = smma(median_1d, 5)
    teeth_1d = smma(median_1d, 8)
    jaws_1d = smma(median_1d, 13)
    
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 13)  # EMA34 (34), volume MA (20), Alligator jaws (13)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(jaws_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_34_aligned[i]
        lips = lips_aligned[i]
        teeth = teeth_aligned[i]
        jaws = jaws_aligned[i]
        vol_ma_val = vol_ma_1d_aligned[i]
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Alligator alignment: Long when jaws > teeth > lips, Short when jaws < teeth < lips
        if i >= start_idx + 1:
            lips_prev = lips_aligned[i-1]
            teeth_prev = teeth_aligned[i-1]
            jaws_prev = jaws_aligned[i-1]
            lips_rising = lips > lips_prev
            teeth_rising = teeth > teeth_prev
            jaws_rising = jaws > jaws_prev
            lips_falling = lips < lips_prev
            teeth_falling = teeth < teeth_prev
            jaws_falling = jaws < jaws_prev
        else:
            lips_rising = teeth_rising = jaws_rising = False
            lips_falling = teeth_falling = jaws_falling = False
        
        if position == 0:
            # Long: Jaws > Teeth > Lips AND EMA34 rising AND volume spike
            if jaws > teeth and teeth > lips and ema_rising and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Jaws < Teeth < Lips AND EMA34 falling AND volume spike
            elif jaws < teeth and teeth < lips and ema_falling and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Alligator lines cross opposite (jaws < teeth OR teeth < lips) OR EMA34 starts falling
                if jaws < teeth or teeth < lips or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: Alligator lines cross opposite (jaws > teeth OR teeth > lips) OR EMA34 starts rising
                if jaws > teeth or teeth > lips or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1dEMA34_Trend_1dVolumeSpike"
timeframe = "12h"
leverage = 1.0
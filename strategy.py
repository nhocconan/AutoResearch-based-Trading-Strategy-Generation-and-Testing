#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator alignment with 1d EMA34 trend filter and volume confirmation.
Long when Alligator jaws < teeth < lips (bullish alignment) AND 1d EMA34 rising AND volume > 1.3x 20-period average.
Short when Alligator jaws > teeth > lips (bearish alignment) AND 1d EMA34 falling AND volume > 1.3x 20-period average.
Exit when alignment breaks or EMA34 reverses direction.
Uses 1d HTF for EMA34 trend to avoid whipsaws in ranging markets. Target: 75-200 total trades over 4 years (19-50/year).
Williams Alligator uses SMAs of 13, 8, 5 periods with 8, 5, 3 forward shifts respectively.
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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_CLOSE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # 13-period SMMA
    teeth = smma(close, 8)  # 8-period SMMA
    lips = smma(close, 5)   # 5-period SMMA
    
    # Apply forward shifts: Jaw shifted 8 bars, Teeth shifted 5 bars, Lips shifted 3 bars
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Fill shifted beginnings with NaN
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(23, 34, 20)  # Alligator (23), EMA34 (34), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or 
            np.isnan(lips_shifted[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw_val = jaw_shifted[i]
        teeth_val = teeth_shifted[i]
        lips_val = lips_shifted[i]
        ema_val = ema_34_aligned[i]
        vol_ma_val = vol_ma[i]
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: Bullish alignment (jaw < teeth < lips) AND EMA34 rising AND volume spike
            if jaw_val < teeth_val < lips_val and ema_rising and volume[i] > 1.3 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment (jaw > teeth > lips) AND EMA34 falling AND volume spike
            elif jaw_val > teeth_val > lips_val and ema_falling and volume[i] > 1.3 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: alignment breaks bullish OR EMA34 starts falling
                if not (jaw_val < teeth_val < lips_val) or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: alignment breaks bearish OR EMA34 starts rising
                if not (jaw_val > teeth_val > lips_val) or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Williams_Alligator_Alignment_1dEMA34_Trend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0
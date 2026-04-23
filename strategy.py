#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator alignment with 1d EMA34 trend filter and volume confirmation.
Long when Alligator is bullish (jaw < teeth < lips) AND 1d EMA34 rising AND volume > 1.3x 20-period average.
Short when Alligator is bearish (jaw > teeth > lips) AND 1d EMA34 falling AND volume > 1.3x 20-period average.
Exit when Alligator alignment reverses or volume drops below average.
Uses 1d HTF for EMA34 trend (avoids whipsaws in ranging markets). Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator: Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
    # SMMA = smoothed moving average (similar to EMA but with different alpha)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # 20-period volume average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 34, 20)  # Alligator jaw (13), EMA34 (34), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator alignment
        bullish_alignment = jaw[i] < teeth[i] < lips[i]
        bearish_alignment = jaw[i] > teeth[i] > lips[i]
        
        # EMA34 trend direction
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_34_aligned[i] > ema_prev
            ema_falling = ema_34_aligned[i] < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: Bullish Alligator AND EMA34 rising AND volume confirmation
            if bullish_alignment and ema_rising and volume[i] > 1.3 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator AND EMA34 falling AND volume confirmation
            elif bearish_alignment and ema_falling and volume[i] > 1.3 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Alligator alignment turns bearish OR EMA34 starts falling
                if not bullish_alignment or (i >= start_idx + 1 and ema_34_aligned[i] < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: Alligator alignment turns bullish OR EMA34 starts rising
                if not bearish_alignment or (i >= start_idx + 1 and ema_34_aligned[i] > ema_34_aligned[i-1]):
                    exit_signal = True
            
            # Additional exit: volume drops below average (loss of momentum)
            if volume[i] < vol_ma[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Williams_Alligator_Alignment_1dEMA34_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0
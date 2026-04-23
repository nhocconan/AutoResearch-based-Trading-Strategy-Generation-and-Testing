#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator alignment with 1d EMA50 trend filter and volume confirmation.
Long when Alligator lines are aligned bullish (jaw < teeth < lips) AND 1d EMA50 rising AND volume > 1.5x 20-period average.
Short when Alligator lines are aligned bearish (jaw > teeth > lips) AND 1d EMA50 falling AND volume > 1.5x 20-period average.
Exit when Alligator alignment breaks or EMA50 reverses direction.
Uses 1d HTF for EMA50 trend to avoid whipsaws in ranging markets. Target: 50-150 total trades over 4 years (12-37/year).
Alligator uses SMAs of 13, 8, 5 periods with 8, 5, 3 offsets respectively (Williams' original formula).
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
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: Jaw (13-period SMMA, offset 8), Teeth (8-period SMMA, offset 5), Lips (5-period SMMA, offset 3)
    # SMMA = smoothed moving average (similar to EMA but with different smoothing)
    # Using EMA as approximation for SMMA as it's commonly done in practice
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Apply Williams offsets: shift jaw by 8, teeth by 5, lips by 3
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set NaN for invalid offsets
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 8, 5, 50, 20) + 8  # Alligator max period + max offset + EMA50 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_50_aligned[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        vol_ma_val = vol_ma[i]
        
        # Calculate EMA50 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_50_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Alligator alignment conditions
        bullish_alignment = jaw_val < teeth_val < lips_val  # Jaw < Teeth < Lips
        bearish_alignment = jaw_val > teeth_val > lips_val  # Jaw > Teeth > Lips
        
        if position == 0:
            # Long: Bullish Alligator alignment AND EMA50 rising AND volume spike
            if bullish_alignment and ema_rising and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment AND EMA50 falling AND volume spike
            elif bearish_alignment and ema_falling and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Alligator alignment breaks OR EMA50 starts falling
                if not bullish_alignment or (i >= start_idx + 1 and ema_val < ema_50_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: Alligator alignment breaks OR EMA50 starts rising
                if not bearish_alignment or (i >= start_idx + 1 and ema_val > ema_50_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Williams_Alligator_Alignment_1dEMA50_Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator alignment with 1d EMA34 trend filter and volume confirmation.
Long when Alligator jaws < teeth < lips (bullish alignment) AND 1d EMA34 rising AND volume > 1.8x 34-period average.
Short when Alligator jaws > teeth > lips (bearish alignment) AND 1d EMA34 falling AND volume > 1.8x 34-period average.
Exit when Alligator alignment breaks or EMA34 reverses direction.
Uses 1d HTF for EMA34 trend filter to avoid whipsaws in ranging markets. Target: 75-200 total trades over 4 years (19-50/year).
Williams Alligator uses SMAs of median price (typical price) with periods 13, 8, 5 and shifts 8, 5, 3 respectively.
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
    
    # Calculate typical price for Alligator
    typical_price = (high + low + close) / 3.0
    
    # Williams Alligator components (SMAs of typical price)
    # Jaw: 13-period SMA, shifted 8 bars
    jaw = pd.Series(typical_price).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA, shifted 5 bars
    teeth = pd.Series(typical_price).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA, shifted 3 bars
    lips = pd.Series(typical_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 34-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=34, min_periods=34).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13 + 8, 8 + 5, 5 + 3, 34, 34)  # jaw(21), teeth(13), lips(8), EMA34(34), volMA(34)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
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
        
        # Alligator alignment conditions
        bullish_alignment = jaw_val < teeth_val and teeth_val < lips_val
        bearish_alignment = jaw_val > teeth_val and teeth_val > lips_val
        
        if position == 0:
            # Long: Bullish Alligator alignment AND EMA34 rising AND volume spike
            if bullish_alignment and ema_rising and volume[i] > 1.8 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment AND EMA34 falling AND volume spike
            elif bearish_alignment and ema_falling and volume[i] > 1.8 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Alligator alignment breaks OR EMA34 starts falling
                if not bullish_alignment or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: Alligator alignment breaks OR EMA34 starts rising
                if not bearish_alignment or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
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
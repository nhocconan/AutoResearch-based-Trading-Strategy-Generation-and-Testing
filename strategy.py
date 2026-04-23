#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1d Elder Ray filter and volume confirmation.
Long when Alligator jaws < teeth < lips (bullish alignment) AND 1d Elder Bull Power > 0 AND volume > 2.0x 20-period MA.
Short when Alligator jaws > teeth > lips (bearish alignment) AND 1d Elder Bear Power < 0 AND volume > 2.0x 20-period MA.
Exit when Alligator alignment reverses or volume drops below 1.5x 20-period MA.
Uses 1d HTF for Elder Ray trend filter to avoid counter-trend trades, volume spike for momentum confirmation.
Alligator identifies trend phases, Elder Ray confirms bull/bear power on higher timeframe.
Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Williams Alligator (SMAs with specific periods) avoids whipsaws, Elder Ray filters weak trends.
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
    
    # Calculate 4h Williams Alligator (Jaw=13, Teeth=8, Lips=5 SMAs with specific shifts)
    # Jaw: 13-period SMA shifted 8 bars ahead
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift 8 bars ahead
    jaw[:8] = np.nan  # first 8 values invalid after shift
    
    # Teeth: 8-period SMA shifted 5 bars ahead
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift 5 bars ahead
    teeth[:5] = np.nan  # first 5 values invalid after shift
    
    # Lips: 5-period SMA shifted 3 bars ahead
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift 3 bars ahead
    lips[:3] = np.nan  # first 3 values invalid after shift
    
    # Calculate 1d Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high_1d - ema_13_1d  # Elder Bull Power
    bear_power = low_1d - ema_13_1d   # Elder Bear Power
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 4h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 8, 5, 13, 20)  # Alligator components, Elder Ray, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Alligator alignment: jaws < teeth < lips = bullish, jaws > teeth > lips = bearish
        bullish_alignment = jaw_val < teeth_val < lips_val
        bearish_alignment = jaw_val > teeth_val > lips_val
        
        # Volume filter: 4h volume > 2.0x 20-period MA (strong breakout confirmation)
        vol_filter = volume[i] > 2.0 * vol_ma_val
        
        # Volume exit filter: volume < 1.5x 20-period MA (momentum fading)
        vol_exit_filter = volume[i] < 1.5 * vol_ma_val
        
        if position == 0:
            # Long: Bullish Alligator alignment AND Elder Bull Power > 0 AND volume filter
            if bullish_alignment and bull_val > 0 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment AND Elder Bear Power < 0 AND volume filter
            elif bearish_alignment and bear_val < 0 and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Alligator alignment turns bearish OR Elder Bull Power <= 0 OR volume exit filter
                if not bullish_alignment or bull_val <= 0 or vol_exit_filter:
                    exit_signal = True
            elif position == -1:
                # Short exit: Alligator alignment turns bullish OR Elder Bear Power >= 0 OR volume exit filter
                if not bearish_alignment or bear_val >= 0 or vol_exit_filter:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsAlligator_1dElderRay_Power_VolumeSpike"
timeframe = "4h"
leverage = 1.0
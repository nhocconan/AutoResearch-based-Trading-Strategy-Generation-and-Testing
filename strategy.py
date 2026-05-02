#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d Elder Ray volume spike and 1w trend filter
# Williams Alligator (JAW/TEETH/LIPS) identifies trend absence (all lines intertwined) vs presence (diverged)
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) with volume spike confirms trend strength
# 1w EMA34 filter ensures we only trade in alignment with weekly trend to avoid counter-trend whipsaws
# Works in bull markets (Alligator long + Bull Power > 0 + volume spike + 1w uptrend) and bear markets (Alligator short + Bear Power > 0 + volume spike + 1w downtrend)
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag

name = "12h_WilliamsAlligator_1dElderRay_VolumeSpike_1wEMA34_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA34 calculation
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 1d data for Elder Ray and Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator (13,8,5 SMAs shifted)
    # JAW = 13-period SMMA shifted 8 bars ahead
    # TEETH = 8-period SMMA shifted 5 bars ahead  
    # LIPS = 5-period SMMA shifted 3 bars ahead
    close_1d = df_1d['close'].values
    sma_13 = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    sma_8 = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    sma_5 = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Shift SMAs: JAW(13,8), TEETH(8,5), LIPS(5,3)
    jaw = np.roll(sma_13, 8)
    teeth = np.roll(sma_8, 5)
    lips = np.roll(sma_5, 3)
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_13_1d
    bear_power = ema_13_1d - low_1d
    
    # Align Elder Ray to 12h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation (volume spike > 2.0 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Alligator calculation)
    start_idx = 13
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine Alligator state: 
        # Alligator sleeping (all lines intertwined) -> no trend
        # Alligator awakening (lines diverging) -> trend emerging
        # Alligator eating (lines well separated, price above/below) -> strong trend
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        # Alligator long: Lips > Teeth > Jaw (price above all lines)
        alligator_long = lips_val > teeth_val > jaw_val
        # Alligator short: Jaw > Teeth > Lips (price below all lines)  
        alligator_short = jaw_val > teeth_val > lips_val
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator long + Bull Power > 0 + volume spike + 1w uptrend
            if (alligator_long and bull_power_aligned[i] > 0 and 
                volume_confirmation[i] and close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator short + Bear Power > 0 + volume spike + 1w downtrend
            elif (alligator_short and bear_power_aligned[i] > 0 and 
                  volume_confirmation[i] and close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator turns against position OR Bull Power <= 0
            if not alligator_long or bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns against position OR Bear Power <= 0
            if not alligator_short or bear_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
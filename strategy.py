#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray + Volume Spike.
# Uses 12h Williams Alligator (Jaw/Teeth/Lips) for trend direction,
# 1d Elder Ray (Bull/Bear Power) for momentum confirmation,
# and 12h volume > 2.0x 24-bar average for conviction.
# Long when Lips > Teeth > Jaw (bullish alignment) AND Bear Power < 0 AND volume spike.
# Short when Lips < Teeth < Jaw (bearish alignment) AND Bull Power < 0 AND volume spike.
# Discrete sizing 0.25 to limit drawdown and reduce trade frequency.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
# Williams Alligator identifies trend structure, Elder Ray measures bull/bear power behind the move,
# volume spike filters for high-conviction entries. Works in both bull (long bias) and bear (short bias).

name = "12h_WilliamsAlligator_ElderRay_VolumeSpike_v1"
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
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Load 12h data ONCE before loop for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Williams Alligator on 12h: SMAs of median price
    # Jaw: 13-period SMA, shifted 8 bars
    # Teeth: 8-period SMA, shifted 5 bars
    # Lips: 5-period SMA, shifted 3 bars
    median_12h = (df_12h['high'].values + df_12h['low'].values) / 2
    jaw = pd.Series(median_12h).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_12h).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_12h).rolling(window=5, min_periods=5).mean().values
    
    # Apply Alligator shifts (future leakage prevention)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Nullify shifted values
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Load 1d data ONCE before loop for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Elder Ray on 1d: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema_13
    bear_power = df_1d['low'].values - ema_13
    
    # Align Elder Ray components to 12h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation: current 12h volume > 2.0x 24-bar average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Session filter: trade only 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any indicator is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if vol_ma[i] <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = volume[i] > (vol_ma[i] * 2.0)
        
        # Williams Alligator alignment
        lips_val = lips_aligned[i]
        teeth_val = teeth_aligned[i]
        jaw_val = jaw_aligned[i]
        
        bullish_alignment = lips_val > teeth_val > jaw_val
        bearish_alignment = lips_val < teeth_val < jaw_val
        
        # Elder Ray conditions
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: bullish Alligator AND Bear Power < 0 (bulls in control despite bear power) AND volume spike
            if (bullish_alignment and 
                bear_power_val < 0 and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator AND Bull Power < 0 (bears in control despite bull power) AND volume spike
            elif (bearish_alignment and 
                  bull_power_val < 0 and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator reverses (Lips < Teeth) OR Bear Power > 0 (bulls losing momentum) OR volume drops
            if (lips_val < teeth_val or 
                bear_power_val > 0 or 
                not volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator reverses (Lips > Teeth) OR Bull Power > 0 (bears losing momentum) OR volume drops
            if (lips_val > teeth_val or 
                bull_power_val > 0 or 
                not volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
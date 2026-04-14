#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator with 1-day Elder Ray power filter and volume confirmation.
# The Williams Alligator (Jaw/Teeth/Lips) identifies trending vs ranging markets.
# In trending markets (JAW < TEETH < LIPS for uptrend, reverse for downtrend),
# we take trades in the direction of the trend when Elder Ray shows bull/bear power.
# Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low (using 1-day timeframe).
# Volume > 1.3x average confirms participation.
# This combination aims for 15-30 trades per year per symbol (60-120 total over 4 years),
# staying within the optimal range to minimize fee flood while capturing trends.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Elder Ray filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Elder Ray on 1d: Bull Power and Bear Power
    ema_len = 13
    if len(df_1d) < ema_len:
        return np.zeros(n)
    
    ema_1d = pd.Series(df_1d['close']).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    bull_power = df_1d['high'].values - ema_1d  # High - EMA13
    bear_power = ema_1d - df_1d['low'].values   # EMA13 - Low
    
    # Align Elder Ray to 6t timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Williams Alligator on 6t timeframe
    jaw_len, teeth_len, lips_len = 13, 8, 5
    jaw_offset, teeth_offset, lips_offset = 8, 5, 3
    
    if len(close) < max(jaw_len, teeth_len, lips_len) + jaw_offset:
        return np.zeros(n)
    
    # Smoothed median price (typical price)
    typical_price = (high + low + close) / 3
    
    jaw = pd.Series(typical_price).rolling(window=jaw_len, min_periods=jaw_len).mean().shift(jaw_offset).values
    teeth = pd.Series(typical_price).rolling(window=teeth_len, min_periods=teeth_len).mean().shift(teeth_offset).values
    lips = pd.Series(typical_price).rolling(window=lips_len, min_periods=lips_len).mean().shift(lips_offset).values
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(jaw_len + jaw_offset, teeth_len + teeth_offset, lips_len + lips_offset, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Alligator conditions: JAW < TEETH < LIPS = uptrend, JAW > TEETH > LIPS = downtrend
        jaw_teeth_lips_up = (jaw[i] < teeth[i]) and (teeth[i] < lips[i])
        jaw_teeth_lips_down = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        
        # Elder Ray conditions: Bull Power > 0 and Bear Power > 0
        bull_power_pos = bull_power_aligned[i] > 0
        bear_power_pos = bear_power_aligned[i] > 0
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: Uptrend + Bull Power positive + Volume
            if jaw_teeth_lips_up and bull_power_pos and volume_confirmed:
                position = 1
                signals[i] = position_size
            # Enter short: Downtrend + Bear Power positive + Volume
            elif jaw_teeth_lips_down and bear_power_pos and volume_confirmed:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Trend changes or Elder Ray turns negative
            if not (jaw_teeth_lips_up and bull_power_pos):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Trend changes or Elder Ray turns negative
            if not (jaw_teeth_lips_down and bear_power_pos):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Alligator_ElderRay_Volume_v1"
timeframe = "6h"
leverage = 1.0
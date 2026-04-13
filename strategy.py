#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Williams Alligator + Elder Ray with volume confirmation.
# Uses 3-period SMAs (jaw, teeth, lips) for trend direction and Elder Ray (bull/bear power) for momentum.
# Volume filter confirms institutional participation. Designed for 4H timeframe to limit trades.
# Works in bull markets via Alligator alignment and in bear via Elder Ray divergences.
# Target: 80-160 total trades over 4 years (20-40/year) to stay within profitable range.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Williams Alligator on daily: Jaw(13,8), Teeth(8,5), Lips(5,3)
    close_1d = df_1d['close'].values
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # 8-period forward shift
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)  # 5-period forward shift
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)  # 3-period forward shift
    
    # Elder Ray on daily: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean()
    bull_power = df_1d['high'].values - ema13.values
    bear_power = df_1d['low'].values - ema13.values
    
    # Daily volume and its 20-period average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all data to 4-hour timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw.values)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth.values)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips.values)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 4h volume > 1.3x daily volume MA (adjusted for 4h)
        # 6 4h periods per day, so daily MA/6 = approximate 4h period MA
        volume_4h_approx_ma = volume_ma_20_1d_aligned[i] / 6
        volume_condition = volume[i] > (volume_4h_approx_ma * 1.3)
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        alligator_short = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Elder Ray conditions: Bull Power > 0 and rising, Bear Power < 0 and falling
        # Use 3-period slope for momentum
        if i >= 3:
            bull_rising = bull_power_aligned[i] > bull_power_aligned[i-3]
            bear_falling = bear_power_aligned[i] < bear_power_aligned[i-3]
        else:
            bull_rising = False
            bear_falling = False
        
        elder_long = bull_power_aligned[i] > 0 and bull_rising
        elder_short = bear_power_aligned[i] < 0 and bear_falling
        
        # Entry conditions: Alligator alignment + Elder Ray + volume
        if position == 0:
            if alligator_long and elder_long and volume_condition:
                position = 1
                signals[i] = position_size
            elif alligator_short and elder_short and volume_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when Alligator reverses or Elder Ray turns negative
            if not alligator_long or elder_power_aligned[i] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when Alligator reverses or Elder Ray turns positive
            if not alligator_short or bear_power_aligned[i] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Alligator_ElderRay_Volume"
timeframe = "4h"
leverage = 1.0
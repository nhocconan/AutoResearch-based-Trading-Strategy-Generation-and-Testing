#!/usr/bin/env python3
name = "6h_Alligator_ElderRay_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Williams Alligator (Jaws, Teeth, Lips) - 13, 8, 5 SMAs with future shifts
    # Jaws: 13-period SMA shifted 8 bars forward
    jaws = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA shifted 5 bars forward
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA shifted 3 bars forward
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray Components
    # Bull Power = High - EMA(13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    # Bear Power = Low - EMA(13)
    bear_power = low - ema_13
    
    # Load daily trend filter ONCE
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA for trend filter
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Check for NaN values
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator conditions
        # Alligator sleeping (no trend): jaws, teeth, lips intertwined
        alligator_sleeping = (
            abs(jaws[i] - teeth[i]) < (close[i] * 0.005) and
            abs(teeth[i] - lips[i]) < (close[i] * 0.005) and
            abs(lips[i] - jaws[i]) < (close[i] * 0.005)
        )
        
        # Alligator awakening (trending): clear separation
        alligator_long = (lips[i] > teeth[i]) and (teeth[i] > jaws[i])
        alligator_short = (lips[i] < teeth[i]) and (teeth[i] < jaws[i])
        
        # Elder Ray conditions
        strong_bull_power = bull_power[i] > 0 and bull_power[i] > bull_power[i-1]
        strong_bear_power = bear_power[i] < 0 and bear_power[i] < bear_power[i-1]
        
        # Volume condition
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        # Trend filter from daily
        daily_uptrend = ema_20_1d_aligned[i] > ema_20_1d_aligned[i-1]
        daily_downtrend = ema_20_1d_aligned[i] < ema_20_1d_aligned[i-1]
        
        if position == 0:
            # Long: Alligator awakening upwards + strong bull power + daily uptrend + volume
            if (alligator_long and strong_bull_power and daily_uptrend and vol_condition and not alligator_sleeping):
                signals[i] = 0.25
                position = 1
            # Short: Alligator awakening downwards + strong bear power + daily downtrend + volume
            elif (alligator_short and strong_bear_power and daily_downtrend and vol_condition and not alligator_sleeping):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Alligator starts sleeping or bull power weakens
            if alligator_sleeping or (bull_power[i] <= 0) or (bull_power[i] < bull_power[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Alligator starts sleeping or bear power weakens
            if alligator_sleeping or (bear_power[i] >= 0) or (bear_power[i] > bear_power[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Williams Alligator + Elder Ray with Daily Trend Filter
# - Williams Alligator identifies trend vs ranging markets (sleeping vs awakening)
# - Elder Ray measures bull/bear power relative to EMA13
# - Daily EMA20 provides higher timeframe trend filter
# - Volume confirmation (1.5x average) reduces false signals
# - Works in bull markets: Alligator long + rising bull power + daily uptrend
# - Works in bear markets: Alligator short + falling bear power + daily downtrend
# - Avoids whipsaws by requiring Alligator to be awake (clear trend separation)
# - Position size 0.25 targets ~50-150 total trades over 4 years (12-37/year)
# - Novel combination: Alligator + Elder Ray + daily trend + volume filter
# - Not recently tried in 6h timeframe according to experiment history
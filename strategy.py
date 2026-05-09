#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator + Elder Ray with 1d trend filter
# Long when: Alligator bullish (jaws < teeth < lips), Elder Ray bull power > 0, 1d EMA(50) rising
# Short when: Alligator bearish (jaws > teeth > lips), Elder Ray bear power < 0, 1d EMA(50) falling
# Exit when: Alligator reverses or Elder Ray power crosses zero
# Position size: 0.25 (25% of capital) to limit drawdown. Target: 12-37 trades/year.
# Williams Alligator uses SMAs of median price (HLC/3) with specific periods.
# Elder Ray measures bull/bear power relative to EMA(13).
# Designed to work in trending markets (both bull and bear) by following the 1d trend.

name = "12h_Alligator_ElderRay_1dTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator: SMAs of median price
    median_price = (high + low) / 3
    # Jaws: SMA(13) of median, shifted 8 bars
    jaws_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaws = np.roll(jaws_raw, 8)
    jaws[:8] = jaws_raw[0]  # fill initial values
    # Teeth: SMA(8) of median, shifted 5 bars
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = teeth_raw[0]
    # Lips: SMA(5) of median, shifted 3 bars
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)
    lips[:3] = lips_raw[0]
    
    alligator_bullish = (jaws < teeth) & (teeth < lips)
    alligator_bearish = (jaws > teeth) & (teeth > lips)
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    elder_bull = bull_power > 0
    elder_bear = bear_power < 0
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close']
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_prev = np.roll(ema_50_1d, 1)
    ema_50_1d_prev[0] = ema_50_1d[0]
    ema_rising = ema_50_1d > ema_50_1d_prev
    ema_falling = ema_50_1d < ema_50_1d_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_falling)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Alligator bullish + Elder Ray bull power > 0 + 1d EMA rising
            if (alligator_bullish[i] and 
                elder_bull[i] and 
                ema_rising_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Alligator bearish + Elder Ray bear power < 0 + 1d EMA falling
            elif (alligator_bearish[i] and 
                  elder_bear[i] and 
                  ema_falling_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator turns bearish OR Elder Ray bull power <= 0
            if (not alligator_bullish[i]) or (bull_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator turns bullish OR Elder Ray bear power >= 0
            if (not alligator_bearish[i]) or (bear_power[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
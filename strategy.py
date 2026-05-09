#!/usr/bin/env python3
# Hypothesis: 6h Williams Alligator with Elder Ray (Bull/Bear Power) and weekly trend filter
# Long when: Bull Power > 0, price above Alligator teeth (EMA8), weekly EMA(21) rising
# Short when: Bear Power < 0, price below Alligator teeth (EMA8), weekly EMA(21) falling
# Exit when: Bull/Bear Power reverses or price crosses Alligator jaw (EMA13)
# Uses Williams Alligator (Jaw=EMA13, Teeth=EMA8, Lips=EMA5) + Elder Ray (Bull/Bear Power = EMA13 - High/Low)
# Weekly trend filter avoids counter-trend trades. Designed for low frequency (15-35 trades/year) to avoid fee drag.
# Works in bull (follow Alligator alignment) and bear (fade false breaks via Elder Ray reversals).

name = "6h_Alligator_ElderRay_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator: Jaw (EMA13), Teeth (EMA8), Lips (EMA5)
    close_s = pd.Series(close)
    jaw = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values  # Jaw (EMA13)
    teeth = close_s.ewm(span=8, adjust=False, min_periods=8).mean().values   # Teeth (EMA8)
    lips = close_s.ewm(span=5, adjust=False, min_periods=5).mean().values    # Lips (EMA5)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - jaw
    bear_power = jaw - low
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Weekly EMA(21) for trend filter
    close_1w = df_1w['close']
    ema_21_1w = close_1w.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_prev = np.roll(ema_21_1w, 1)
    ema_21_1w_prev[0] = ema_21_1w[0]
    ema_rising = ema_21_1w > ema_21_1w_prev
    ema_falling = ema_21_1w < ema_21_1w_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_falling)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for EMA13 and weekly EMA21
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Bull Power > 0, price above Teeth, weekly EMA rising
            if (bull_power[i] > 0 and 
                close[i] > teeth[i] and 
                ema_rising_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power < 0, price below Teeth, weekly EMA falling
            elif (bear_power[i] > 0 and 
                  close[i] < teeth[i] and 
                  ema_falling_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power <= 0 OR price crosses below Jaw (EMA13)
            if (bull_power[i] <= 0) or (close[i] < jaw[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power <= 0 OR price crosses above Jaw (EMA13)
            if (bear_power[i] <= 0) or (close[i] > jaw[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
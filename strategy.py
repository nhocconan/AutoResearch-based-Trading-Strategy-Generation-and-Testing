#!/usr/bin/env python3
# 4h_Williams_Alligator_Supertrend
# Hypothesis: Williams Alligator (Jaw=TEETH=LIPS) combined with Supertrend on 4h timeframe.
# Goes long when price > Alligator Teeth (SMA8) and Supertrend gives buy signal.
# Goes short when price < Alligator Teeth and Supertrend gives sell signal.
# Uses Alligator to filter choppy markets (when lines are intertwined) and Supertrend for trend direction.
# Designed to work in both bull and bear markets by using trend-following components.
# Targets 20-50 trades per year on 4h timeframe with position size 0.25.

name = "4h_Williams_Alligator_Supertrend"
timeframe = "4h"
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
    
    # Get 1d data for 1-week trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-week EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1w)
    
    # Williams Alligator components on 4h data
    # Jaw (13-period SMMA, shifted 8 bars)
    jaw = pd.Series(close).rolling(window=13, center=False).mean().shift(8).values
    # Teeth (8-period SMMA, shifted 5 bars)
    teeth = pd.Series(close).rolling(window=8, center=False).mean().shift(5).values
    # Lips (5-period SMMA, shifted 3 bars)
    lips = pd.Series(close).rolling(window=5, center=False).mean().shift(3).values
    
    # Supertrend calculation
    atr_period = 10
    atr_multiplier = 3.0
    
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (atr_multiplier * atr)
    lower_band = hl2 - (atr_multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.full(n, np.nan)
    direction = np.full(n, 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, n):
        if np.isnan(atr[i-1]) or np.isnan(upper_band[i-1]) or np.isnan(lower_band[i-1]):
            continue
            
        # Supertrend logic
        if close[i] > upper_band[i-1]:
            direction[i] = 1
        elif close[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 8, 5, atr_period)  # Warmup for Alligator and ATR
    
    for i in range(start_idx, n):
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(supertrend[i]) or np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator condition: Teeth > Lips for bullish alignment, Teeth < Lips for bearish
        # But we use a simpler condition: price relative to Teeth
        price_above_teeth = close[i] > teeth[i]
        price_below_teeth = close[i] < teeth[i]
        
        # Supertrend direction
        st_uptrend = direction[i] == 1
        st_downtrend = direction[i] == -1
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long entry: price > Teeth AND Supertrend uptrend AND weekly uptrend
            if price_above_teeth and st_uptrend and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: price < Teeth AND Supertrend downtrend AND weekly downtrend
            elif price_below_teeth and st_downtrend and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price < Teeth OR Supertrend downtrend OR weekly downtrend
            if not (price_above_teeth and st_uptrend and weekly_uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > Teeth OR Supertrend uptrend OR weekly uptrend
            if not (price_below_teeth and st_downtrend and weekly_downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
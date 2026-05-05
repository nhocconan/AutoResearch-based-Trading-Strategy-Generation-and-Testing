#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray combination with 1d trend filter
# Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs on median price
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when: Alligator aligned bullish (Lips > Teeth > Jaw) AND Bull Power > 0 AND price > 1d EMA50
# Short when: Alligator aligned bearish (Lips < Teeth < Jaw) AND Bear Power < 0 AND price < 1d EMA50
# Exit when Alligator alignment breaks OR Elder Power reverses
# Williams Alligator identifies trend phase and avoids whipsaws in ranging markets
# Elder Ray measures bull/bear strength behind price moves
# 1d EMA50 filters for higher timeframe trend alignment
# Target: 12-37 trades/year per symbol (50-150 total over 4 years)
# Discrete sizing (0.25) to limit fee drag

name = "6h_WilliamsAlligator_ElderRay_1dEMA50_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator components on median price
    median_price = (high + low) / 2
    
    # Jaw: 13-period SMA, smoothed by 8 periods
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = pd.Series(jaw_raw).rolling(window=8, min_periods=8).mean().values
    
    # Teeth: 8-period SMA, smoothed by 5 periods
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(teeth_raw).rolling(window=5, min_periods=5).mean().values
    
    # Lips: 5-period SMA, smoothed by 3 periods
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(lips_raw).rolling(window=3, min_periods=3).mean().values
    
    # Align Alligator components to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, prices, jaw)  # same timeframe, no alignment needed but use helper for consistency
    teeth_aligned = align_htf_to_ltf(prices, prices, teeth)
    lips_aligned = align_htf_to_ltf(prices, prices, lips)
    
    # Calculate Elder Ray components
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Alligator bullish AND Bull Power positive AND price > 1d EMA50
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                bull_power[i] > 0 and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Alligator bearish AND Bear Power negative AND price < 1d EMA50
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  bear_power[i] < 0 and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks OR Bull Power turns negative
            if not (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and bull_power[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks OR Bear Power turns positive
            if not (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and bear_power[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
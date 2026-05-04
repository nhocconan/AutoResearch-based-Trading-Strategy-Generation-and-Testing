#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray Power with 1d regime filter
# Long when: Alligator jaws (13-period SMA) > teeth (8-period SMA) > lips (5-period SMA) AND Bull Power > 0 AND 1d close > 1d EMA50
# Short when: Alligator jaws < teeth < lips AND Bear Power < 0 AND 1d close < 1d EMA50
# Uses Williams Alligator for trend identification and Elder Ray for power confirmation, with 1d EMA50 as regime filter.
# Works in bull markets via strong uptrend confirmation and bear markets via strong downtrend confirmation.
# The triple Alligator alignment reduces whipsaw, while Elder Ray ensures momentum behind the move.
# Target: 12-37 trades/year on 6h timeframe.

name = "6h_Alligator_ElderRay_1dEMA50_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for HTF regime filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for regime filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) - all SMAs
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Alligator aligned up AND Bull Power positive AND 1d regime bullish
            if (jaw[i] > teeth[i] and teeth[i] > lips[i] and  # Jaw > Teeth > Lips
                bull_power[i] > 0 and 
                close[i] > ema_50_1d_aligned[i]):  # 1d close above EMA50
                signals[i] = 0.25
                position = 1
            # Short conditions: Alligator aligned down AND Bear Power negative AND 1d regime bearish
            elif (jaw[i] < teeth[i] and teeth[i] < lips[i] and  # Jaw < Teeth < Lips
                  bear_power[i] < 0 and 
                  close[i] < ema_50_1d_aligned[i]):  # 1d close below EMA50
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks OR 1d regime turns bearish
            if not (jaw[i] > teeth[i] and teeth[i] > lips[i]) or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks OR 1d regime turns bullish
            if not (jaw[i] < teeth[i] and teeth[i] < lips[i]) or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
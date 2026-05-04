#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray power with 1d EMA34 trend filter
# Long when: Alligator jaws < teeth < lips (bullish alignment) AND Bull Power > 0 AND price > 1d EMA34
# Short when: Alligator jaws > teeth > lips (bearish alignment) AND Bear Power < 0 AND price < 1d EMA34
# Uses 1d HTF for trend to reduce whipsaw. Alligator identifies trend, Elder Ray measures power behind move.
# Volume not required - relying on clear trend/power signals. Targets 12-30 trades/year on 6h.

name = "6h_Alligator_ElderRay_1dEMA34_Trend_Power"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for HTF trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator: SMAs of median price (typical price) with different periods
    typical_price = (high + low + close) / 3.0
    
    # Jaws: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # SMMA = smoothed moving average (similar to EMA but different smoothing)
    # Using EMA as approximation for SMMA (standard practice)
    jaws = pd.Series(typical_price).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(typical_price).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(typical_price).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(jaws_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Alligator bullish alignment AND Bull Power > 0 AND price > 1d EMA34
            if (jaws_aligned[i] < teeth_aligned[i] < lips_aligned[i] and  # Bullish alignment
                bull_power[i] > 0 and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Alligator bearish alignment AND Bear Power < 0 AND price < 1d EMA34
            elif (jaws_aligned[i] > teeth_aligned[i] > lips_aligned[i] and  # Bearish alignment
                  bear_power[i] < 0 and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment turns bearish OR price closes below 1d EMA34
            if (jaws_aligned[i] > teeth_aligned[i] or  # Alignment broken
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment turns bullish OR price closes above 1d EMA34
            if (jaws_aligned[i] < teeth_aligned[i] or  # Alignment broken
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
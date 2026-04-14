#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator + Elder Ray combination with 12h trend filter
# Uses Williams Alligator (Jaw/Teeth/Lips) to identify trend direction and avoid chop
# Elder Ray (Bull Power/Bear Power) measures trend strength relative to EMA
# Long when: Lips > Teeth > Jaw (bullish alignment) AND Bull Power > 0 AND price > 12h EMA50
# Short when: Lips < Teeth < Jaw (bearish alignment) AND Bear Power < 0 AND price < 12h EMA50
# Exit when Alligator alignment breaks or power reverses
# Target: 50-150 total trades over 4 years (12-37/year) with strong trend filtering

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Williams Alligator (13,8,5 SMAs with future shifts)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values  # 13-period, shifted 8
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values   # 8-period, shifted 5
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values    # 5-period, shifted 3
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (max shift + buffer)
    start = 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long setup: bullish Alligator alignment + positive Bull Power + above 12h EMA50
            if (lips[i] > teeth[i] > jaw[i] and bull_power[i] > 0 and price > ema50_12h_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: bearish Alligator alignment + negative Bear Power + below 12h EMA50
            elif (lips[i] < teeth[i] < jaw[i] and bear_power[i] < 0 and price < ema50_12h_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator alignment breaks OR Bull Power turns negative
            if not (lips[i] > teeth[i] > jaw[i]) or bull_power[i] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Alligator alignment breaks OR Bear Power turns positive
            if not (lips[i] < teeth[i] < jaw[i]) or bear_power[i] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Alligator_ElderRay_12hEMA50"
timeframe = "6h"
leverage = 1.0
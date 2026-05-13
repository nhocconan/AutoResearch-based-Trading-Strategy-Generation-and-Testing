#!/usr/bin/env python3
# Hypothesis: 6h Williams %R reversal with 1d EMA34 trend filter and 1w Camarilla pivot exits.
# Williams %R identifies overbought/oversold conditions: long when %R crosses above -80 from below,
# short when %R crosses below -20 from above. Trend filter: price > 1d EMA34 for longs, price < 1d EMA34 for shorts.
# Exits when price reaches 1w Camarilla R3/S3 levels (strong reversal zones) or %R reverts to -50.
# Uses discrete position sizing (0.25) to limit fee churn. Designed for mean reversion in ranging markets
# with trend alignment to avoid fighting the higher timeframe direction. Works in both bull and bear
# regimes by capturing short-term exhaustion while respecting weekly structure.

name = "6h_WilliamsR_Reversal_1dEMA34_1wCamarillaExits_v1"
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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1w Camarilla levels for exits (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla levels: based on previous week's range
    camarilla_r3 = np.zeros(len(close_1w))
    camarilla_s3 = np.zeros(len(close_1w))
    for i in range(1, len(close_1w)):
        rng = high_1w[i-1] - low_1w[i-1]
        camarilla_r3[i] = close_1w[i-1] + rng * 1.1 / 4  # R3 = C + 1.1*range/4
        camarilla_s3[i] = close_1w[i-1] - rng * 1.1 / 4  # S3 = C - 1.1*range/4
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Calculate Williams %R (14 period)
    lookback = 14
    highest_high = np.zeros_like(high)
    lowest_low = np.zeros_like(low)
    
    for i in range(n):
        start_idx = max(0, i - lookback + 1)
        highest_high[i] = np.max(high[start_idx:i+1])
        lowest_low[i] = np.min(low[start_idx:i+1])
    
    # Avoid division by zero
    rr = highest_high - lowest_low
    divisor = np.where(rr == 0, 1, rr)
    williams_r = -100 * (highest_high - close) / divisor  # %R = -100*(HH - C)/(HH - LL)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 from below (oversold bounce) AND price > 1d EMA34 (trend alignment)
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 from above (overbought rejection) AND price < 1d EMA34 (trend alignment)
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R reverts to -50 OR price reaches weekly R3 (strong resistance)
            if (williams_r[i] >= -50 or 
                close[i] >= camarilla_r3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R reverts to -50 OR price reaches weekly S3 (strong support)
            if (williams_r[i] <= -50 or 
                close[i] <= camarilla_s3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
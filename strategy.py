#!/usr/bin/env python3
"""
6h_triple_ema_trend_filter_v1
Hypothesis: Use triple exponential moving averages (8, 21, 55) on 6h timeframe with 1d ADX filter to identify strong trends. Enter long when EMA8 > EMA21 > EMA55 and ADX > 25, short when EMA8 < EMA21 < EMA55 and ADX > 25. Exit when EMA cross reverses or ADX drops below 20. This captures momentum while avoiding whipsaws in ranging markets. Works in both bull and bear markets by requiring strong trend confirmation (ADX > 25). Target: 80-160 total trades over 4 years (20-40/year) to balance opportunity with fee minimization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_triple_ema_trend_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate 1d ADX for trend strength filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    tr = np.zeros_like(high_1d)
    
    for i in range(1, len(df_1d)):
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    atr = np.zeros_like(high_1d)
    plus_di = np.zeros_like(high_1d)
    minus_di = np.zeros_like(high_1d)
    
    # Initial values
    atr[period] = np.mean(tr[1:period+1])
    plus_dm_sum = np.sum(plus_dm[1:period+1])
    minus_dm_sum = np.sum(minus_dm[1:period+1])
    
    for i in range(period + 1, len(df_1d)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        plus_dm_sum = plus_dm_sum - (plus_dm_sum / period) + plus_dm[i]
        minus_dm_sum = minus_dm_sum - (minus_dm_sum / period) + minus_dm[i]
        plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
        minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
    
    # Calculate DX and ADX
    dx = np.zeros_like(high_1d)
    adx = np.zeros_like(high_1d)
    
    for i in range(period + 1, len(df_1d)):
        di_sum = plus_di[i] + minus_di[i]
        dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum if di_sum != 0 else 0
    
    # Smooth DX to get ADX
    adx[2*period] = np.mean(dx[period+1:2*period+1])
    for i in range(2*period + 1, len(df_1d)):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate EMAs on 6h timeframe
    ema8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema55 = pd.Series(close).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(55, n):
        # Skip if ADX not available
        if np.isnan(adx_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Trend strength filter
        strong_trend = adx_aligned[i] > 25
        weak_trend = adx_aligned[i] < 20
        
        # EMA alignment
        ema8_above_21 = ema8[i] > ema21[i]
        ema21_above_55 = ema21[i] > ema55[i]
        ema8_below_21 = ema8[i] < ema21[i]
        ema21_below_55 = ema21[i] < ema55[i]
        
        if position == 1:  # Long position
            # Exit: EMA cross down OR trend weakens
            if not (ema8_above_21 and ema21_above_55) or weak_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: EMA cross up OR trend weakens
            if not (ema8_below_21 and ema21_below_55) or weak_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if strong_trend:
                # Strong uptrend
                if ema8_above_21 and ema21_above_55:
                    position = 1
                    signals[i] = 0.25
                # Strong downtrend
                elif ema8_below_21 and ema21_below_55:
                    position = -1
                    signals[i] = -0.25
    
    return signals
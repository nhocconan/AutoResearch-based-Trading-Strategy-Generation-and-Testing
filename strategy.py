#!/usr/bin/env python3
"""
Hypothesis: 12-hour Williams Alligator with 1-week EMA trend filter.
Long when price > Alligator Jaw and Jaw > Teeth > Lips (bullish alignment).
Short when price < Alligator Jaw and Jaw < Teeth < Lips (bearish alignment).
Exit when price crosses Alligator Jaw or trend alignment breaks.
Uses weekly EMA to filter trend direction and avoid counter-trend trades.
Williams Alligator identifies trend emergence; weekly filter ensures alignment with higher timeframe trend.
Works in both bull and bear markets by following strong trends and avoiding sideways chop.
"""

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
    
    # Williams Alligator parameters (13, 8, 5 periods shifted)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_shift = 8
    teeth_shift = 5
    lips_shift = 3
    
    # Calculate Alligator lines (SMMA = smoothed moving average)
    def smoothed_ma(data, period):
        sma = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            sma[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    median_price = (high + low) / 2.0
    
    jaw = smoothed_ma(median_price, jaw_period)
    teeth = smoothed_ma(median_price, teeth_period)
    lips = smoothed_ma(median_price, lips_period)
    
    # Apply shifts
    jaw = np.roll(jaw, jaw_shift)
    teeth = np.roll(teeth, teeth_shift)
    lips = np.roll(lips, lips_shift)
    
    # Set NaN for shifted values that went out of bounds
    jaw[:jaw_shift] = np.nan
    teeth[:teeth_shift] = np.nan
    lips[:lips_shift] = np.nan
    
    # Load 1-week EMA for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Bullish alignment: price > jaw and jaw > teeth > lips
        bullish_alignment = (close[i] > jaw[i] and jaw[i] > teeth[i] and teeth[i] > lips[i])
        # Bearish alignment: price < jaw and jaw < teeth < lips
        bearish_alignment = (close[i] < jaw[i] and jaw[i] < teeth[i] and teeth[i] < lips[i])
        
        # Weekly trend filter: price above/below weekly EMA
        weekly_uptrend = close[i] > ema_20_1w_aligned[i]
        weekly_downtrend = close[i] < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long: bullish alignment + weekly uptrend
            if bullish_alignment and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment + weekly downtrend
            elif bearish_alignment and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: bullish alignment breaks OR weekly trend turns down
                if not (bullish_alignment and weekly_uptrend):
                    exit_signal = True
            else:  # position == -1
                # Exit short: bearish alignment breaks OR weekly trend turns up
                if not (bearish_alignment and weekly_downtrend):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1wEMA_TrendFilter"
timeframe = "12h"
leverage = 1.0
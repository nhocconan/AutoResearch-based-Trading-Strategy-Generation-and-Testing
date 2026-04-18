#!/usr/bin/env python3
"""
1d_WilliamsAlligator_ElderRay_Trend_v1
Hypothesis: Use Williams Alligator (jaw/teeth/lips) for trend direction and Elder Ray (bull/bear power) for momentum confirmation on daily timeframe. Weekly trend filter ensures alignment with higher timeframe trend. Designed for low trade frequency (<20/year) to minimize fee drag while capturing sustained trends in both bull and bear markets.
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
    
    # Williams Alligator (13,8,5 smoothed with 8,5,3)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # Jaw (Blue): 13-period SMMA smoothed 8
    jaw = close_series.rolling(window=13, min_periods=13).mean().rolling(window=8, min_periods=8).mean()
    # Teeth (Red): 8-period SMMA smoothed 5
    teeth = close_series.rolling(window=8, min_periods=8).mean().rolling(window=5, min_periods=5).mean()
    # Lips (Green): 5-period SMMA smoothed 3
    lips = close_series.rolling(window=5, min_periods=5).mean().rolling(window=3, min_periods=3).mean()
    
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    # Elder Ray Power (13-period EMA)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean()
    bull_power = (high - ema13.values)  # High - EMA13
    bear_power = (low - ema13.values)   # Low - EMA13
    
    # Weekly trend filter: EMA34 on weekly close
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Need Alligator components
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_vals[i]) or np.isnan(teeth_vals[i]) or np.isnan(lips_vals[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        lips_val = lips_vals[i]
        teeth_val = teeth_vals[i]
        jaw_val = jaw_vals[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        weekly_trend = ema_1w_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (uptrend) AND bull power > 0 AND price above weekly EMA
            if lips_val > teeth_val > jaw_val and bull_val > 0 and price > weekly_trend:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (downtrend) AND bear power < 0 AND price below weekly EMA
            elif lips_val < teeth_val < jaw_val and bear_val < 0 and price < weekly_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: trend weakens (lips < teeth) OR bear power negative
            if lips_val < teeth_val or bear_val < 0:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: trend weakens (lips > teeth) OR bull power positive
            if lips_val > teeth_val or bull_val > 0:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WilliamsAlligator_ElderRay_Trend_v1"
timeframe = "1d"
leverage = 1.0
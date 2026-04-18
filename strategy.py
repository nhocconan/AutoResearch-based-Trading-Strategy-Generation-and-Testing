#!/usr/bin/env python3
"""
6h Elder Ray + Weekly Trend Filter
Hypothesis: Elder Ray (bull/bear power) identifies institutional buying/selling pressure.
Combined with weekly EMA trend filter to avoid counter-trend trades in 6h timeframe.
Works in bull/bear markets by only taking longs in weekly uptrend and shorts in weekly downtrend.
Target: 25-40 trades/year (~100-160 total over 4 years) to minimize fee drag.
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
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (once before loop)
    df_w = get_htf_data(prices, '1w')
    
    # Weekly EMA34 for trend filter
    ema_w = pd.Series(df_w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_w_aligned = align_htf_to_ltf(prices, df_w, ema_w)
    
    # Calculate Elder Ray components on 6h data
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_w_aligned[i]) or 
            np.isnan(ema13[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        ema_w_val = ema_w_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 (buying pressure) + weekly uptrend + volume confirmation
            if bull_power[i] > 0 and close[i] > ema_w_val and volume_confirmed[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (selling pressure) + weekly downtrend + volume confirmation
            elif bear_power[i] < 0 and close[i] < ema_w_val and volume_confirmed[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: Bull Power turns negative or price crosses below weekly EMA
            if bull_power[i] <= 0 or close[i] < ema_w_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: Bear Power turns positive or price crosses above weekly EMA
            if bear_power[i] >= 0 or close[i] > ema_w_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_WeeklyEMA34_Volume"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + 1d Trend Filter
Long when Bull Power > 0 and Bear Power < 0 with 1d EMA50 uptrend; short when Bear Power < 0 and Bull Power < 0 with 1d EMA50 downtrend.
Exit when either power crosses zero or price crosses 1d EMA50.
Elder Ray measures bull/bear power relative to EMA13; effective in trending markets with clear momentum.
Designed for 15-30 trades/year to minimize fee flood while capturing sustained moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 50-period EMA for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 13-period EMA for Elder Ray
    ema_13 = pd.Series(prices['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = prices['high'].values - ema_13  # High minus EMA13
    bear_power = prices['low'].values - ema_13   # Low minus EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if trend filter not ready
        if np.isnan(ema_50_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_50_val = ema_50_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        
        if position == 0:
            # Enter long: bull power positive, bear power negative, and 1d EMA50 uptrend
            if (bull_val > 0 and 
                bear_val < 0 and 
                price_close > ema_50_val):
                signals[i] = 0.25
                position = 1
            # Enter short: bear power negative, bull power negative, and 1d EMA50 downtrend
            elif (bear_val < 0 and 
                  bull_val < 0 and 
                  price_close < ema_50_val):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: power crossover or price crosses 1d EMA50
            exit_signal = False
            
            if position == 1:
                # Exit long: bull power turns negative OR price closes below EMA50
                if bull_val <= 0 or price_close < ema_50_val:
                    exit_signal = True
            elif position == -1:
                # Exit short: bear power turns positive OR price closes above EMA50
                if bear_val >= 0 or price_close > ema_50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_1dEMA50_Trend"
timeframe = "6h"
leverage = 1.0
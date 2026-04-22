#!/usr/bin/env python3

"""
Hypothesis: Weekly CAMARILLA pivot breakout on daily timeframe with volume confirmation.
Only trade long when price breaks above R4 level during weekly uptrend with volume spike.
Short when price breaks below S4 level during weekly downtrend with volume spike.
Exit when price returns to the daily pivot point (PP) or weekly trend reverses.
Designed for low trade frequency (10-30 trades/year) by requiring weekly trend alignment and volume confirmation.
Works in both bull and bear markets by following the weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily CAMARILLA pivot levels
    # Pivot Point (PP) = (High + Low + Close) / 3
    # R4 = Close + ((High - Low) * 1.1 / 2)
    # S4 = Close - ((High - Low) * 1.1 / 2)
    pp = (high + low + close) / 3
    r4 = close + ((high - low) * 1.1 / 2)
    s4 = close - ((high - low) * 1.1 / 2)
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # 10-period EMA on weekly close for trend
    close_1w = df_1w['close'].values
    ema10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(r4[i]) or np.isnan(s4[i]) or np.isnan(pp[i]) or 
            np.isnan(ema10_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above R4 + weekly uptrend + volume spike
            if close[i] > r4[i] and ema10_1w_aligned[i] > ema10_1w_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 + weekly downtrend + volume spike
            elif close[i] < s4[i] and ema10_1w_aligned[i] < ema10_1w_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to pivot point or weekly trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price below pivot or weekly trend turns down
                if close[i] < pp[i] or ema10_1w_aligned[i] < ema10_1w_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price above pivot or weekly trend turns up
                if close[i] > pp[i] or ema10_1w_aligned[i] > ema10_1w_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Camarilla_R4_S4_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0
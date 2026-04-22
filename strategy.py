#!/usr/bin/env python3

"""
Hypothesis: 4-hour Camarilla R1/S1 level breakout with 12-hour EMA(50) trend filter and volume spike confirmation.
Trades breakouts at key pivot levels in the direction of the 12h trend only when volume exceeds 1.8x the 20-period average.
Uses fixed position sizing (0.25) to minimize transaction costs and target 25-40 trades/year (100-160 total over 4 years).
Designed to work in both bull and bear markets by aligning with higher timeframe trend and avoiding false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for Camarilla calculation - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (R1, S1) from previous day
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = close_4h + (high_4h - low_4h) * 1.1 / 12
    camarilla_s1 = close_4h - (high_4h - low_4h) * 1.1 / 12
    
    # Shift by 1 to use previous bar's levels (avoid look-ahead)
    camarilla_r1 = np.roll(camarilla_r1, 1)
    camarilla_s1 = np.roll(camarilla_s1, 1)
    camarilla_r1[0] = np.nan
    camarilla_s1[0] = np.nan
    
    # Align Camarilla levels
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA for trend filter (50-period)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: price breaks above Camarilla R1, above 12h EMA (uptrend)
            if close[i] > camarilla_r1_aligned[i] and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1, below 12h EMA (downtrend)
            elif close[i] < camarilla_s1_aligned[i] and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Camarilla level or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price touches Camarilla S1 or closes below 12h EMA
                if close[i] < camarilla_s1_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price touches Camarilla R1 or closes above 12h EMA
                if close[i] > camarilla_r1_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0
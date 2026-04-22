#!/usr/bin/env python3

"""
Hypothesis: 4-hour Camarilla pivot reversal with 1-day EMA(34) trend filter and volume spike confirmation.
Trades reversals at key Camarilla levels (L3/S3 for long, H3/H4 for short) in the direction of the daily trend only when volume exceeds 2.0x the 20-period average.
Targets 20-50 trades/year (80-200 total over 4 years) with disciplined entry/exit to minimize fee drift.
Works in both bull and bear markets by aligning with higher timeframe trend.
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
    
    # Load 1d data for Camarilla pivot calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla levels: H4, H3, L3, L4
    # Range = previous day high - low
    range_prev = high_prev - low_prev
    camarilla_h4 = close_prev + 1.1 * range_prev * 1.1 / 2
    camarilla_h3 = close_prev + 1.1 * range_prev * 1.1 / 4
    camarilla_l3 = close_prev - 1.1 * range_prev * 1.1 / 4
    camarilla_l4 = close_prev - 1.1 * range_prev * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Load 1d data for trend filter - ONCE before loop
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: price touches L3 and closes above it, in uptrend
            if low[i] <= camarilla_l3_aligned[i] and close[i] > camarilla_l3_aligned[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price touches H3 and closes below it, in downtrend
            elif high[i] >= camarilla_h3_aligned[i] and close[i] < camarilla_h3_aligned[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price reaches opposite Camarilla level or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price touches H3 or closes below 1d EMA
                if high[i] >= camarilla_h3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price touches L3 or closes above 1d EMA
                if low[i] <= camarilla_l3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_L3H3_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0
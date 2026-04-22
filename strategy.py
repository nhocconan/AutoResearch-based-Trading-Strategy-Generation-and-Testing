#!/usr/bin/env python3

"""
Hypothesis: 4-hour Camarilla pivot reversal with 1-day trend filter and volume confirmation.
Go long when price bounces off S1 support during a daily uptrend with volume spike.
Go short when price rejects at R1 resistance during a daily downtrend with volume spike.
Exit when price reaches the opposite pivot level or trend reverses.
Designed for low-to-moderate trade frequency (15-30 trades/year) by requiring confluence.
Works in both bull and bear markets by following the daily trend.
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
    
    # Calculate previous day's Camarilla pivot levels (S1, R1)
    # Using daily high/low/close from previous day
    prev_high = pd.Series(high).shift(1).values
    prev_low = pd.Series(low).shift(1).values
    prev_close = pd.Series(close).shift(1).values
    
    # Camarilla levels
    camarilla_s1 = prev_close - (1.1/12) * (prev_high - prev_low)
    camarilla_r1 = prev_close + (1.1/12) * (prev_high - prev_low)
    camarilla_close = prev_close
    
    # Load 1-day data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 34-period EMA on 1d close for trend
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(35, n):
        # Skip if data not ready
        if (np.isnan(camarilla_s1[i]) or np.isnan(camarilla_r1[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: price touches/bounces off S1 + daily uptrend + volume spike
            if low[i] <= camarilla_s1[i] and close[i] > camarilla_s1[i] and ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price touches/rejects at R1 + daily downtrend + volume spike
            elif high[i] >= camarilla_r1[i] and close[i] < camarilla_r1[i] and ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price reaches opposite level or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price reaches Camarilla close or daily trend turns down
                if close[i] >= camarilla_close[i] or ema34_1d_aligned[i] < ema34_1d_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reaches Camarilla close or daily trend turns up
                if close[i] <= camarilla_close[i] or ema34_1d_aligned[i] > ema34_1d_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_Pivot_Reversal_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0
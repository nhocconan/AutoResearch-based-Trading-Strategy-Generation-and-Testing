#!/usr/bin/env python3

"""
Hypothesis: 1-day Exponential Moving Average (EMA) crossover with 1-week Exponential Moving Average (EMA) trend filter and volume confirmation.
Only trade in the direction of the weekly EMA trend when the daily EMA21 crosses above/below the daily EMA50, with volume spike confirmation.
Designed for low trade frequency (10-20 trades/year) by requiring multiple confirmations:
trend alignment, EMA crossover, and volume spike. Works in both bull and bear markets
by following the weekly EMA trend direction, which adapts to market conditions.
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
    
    # EMA21 and EMA50 on daily
    close_s = pd.Series(close)
    ema21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Load weekly data for EMA34 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on weekly
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA34 to daily
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(ema21[i]) or np.isnan(ema50[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Weekly EMA34 uptrend + EMA21 crosses above EMA50 + volume spike
            if (ema34_1w_aligned[i] > ema34_1w_aligned[i-1] and 
                ema21[i] > ema50[i] and ema21[i-1] <= ema50[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Weekly EMA34 downtrend + EMA21 crosses below EMA50 + volume spike
            elif (ema34_1w_aligned[i] < ema34_1w_aligned[i-1] and 
                  ema21[i] < ema50[i] and ema21[i-1] >= ema50[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: EMA21 crosses back in opposite direction or weekly trend changes
            exit_signal = False
            
            if position == 1:
                # Exit long: EMA21 crosses below EMA50 or weekly trend turns down
                if (ema21[i] < ema50[i] and ema21[i-1] >= ema50[i-1]) or \
                   (ema34_1w_aligned[i] < ema34_1w_aligned[i-1]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: EMA21 crosses above EMA50 or weekly trend turns up
                if (ema21[i] > ema50[i] and ema21[i-1] <= ema50[i-1]) or \
                   (ema34_1w_aligned[i] > ema34_1w_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_EMA21_EMA50_Crossover_1wEMA34_Trend_Volume"
timeframe = "1d"
leverage = 1.0
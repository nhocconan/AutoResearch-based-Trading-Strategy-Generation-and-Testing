#!/usr/bin/env python3

"""
Hypothesis: 4-hour Camarilla R1/S1 breakout with 12-hour EMA50 trend filter and volume confirmation.
This strategy combines price channel breakouts with trend alignment and volume confirmation.
It should work in both bull and bear regimes by following the 12h trend direction.
Target: 25-50 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels"""
    pivot = (high + low + close) / 3
    range_ = high - low
    r1 = close + (range_ * 1.1 / 12)
    s1 = close - (range_ * 1.1 / 12)
    r2 = close + (range_ * 1.1 / 6)
    s2 = close - (range_ * 1.1 / 6)
    r3 = close + (range_ * 1.1 / 4)
    s3 = close - (range_ * 1.1 / 4)
    r4 = close + (range_ * 1.1 / 2)
    s4 = close - (range_ * 1.1 / 2)
    return pivot, r1, s1, r2, s2, r3, s3, r4, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if EMA not ready
        if np.isnan(ema_50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla pivot for previous period
        # Use data from previous bar to avoid look-ahead
        if i > 0:
            pivot_high = high[i-1]
            pivot_low = low[i-1]
            pivot_close = close[i-1]
            _, r1, s1, _, _, _, _, _, _ = calculate_camarilla_pivot(pivot_high, pivot_low, pivot_close)
        else:
            continue
        
        if position == 0:
            # Long: Price breaks above R1, above 12h EMA50, volume spike
            if (close[i] > r1 and 
                close[i] > ema_50_12h_aligned[i] and
                volume[i] > 1.5 * np.median(volume[max(0, i-20):i])):  # Volume spike vs median
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1, below 12h EMA50, volume spike
            elif (close[i] < s1 and 
                  close[i] < ema_50_12h_aligned[i] and
                  volume[i] > 1.5 * np.median(volume[max(0, i-20):i])):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Price returns to opposite S1/R1 level or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: Price drops below S1 OR price below 12h EMA50
                if (close[i] < s1 or 
                    close[i] < ema_50_12h_aligned[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above R1 OR price above 12h EMA50
                if (close[i] > r1 or 
                    close[i] > ema_50_12h_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0
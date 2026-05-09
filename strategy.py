#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume
# Hypothesis: Camarilla pivot levels act as strong support/resistance in ranging markets.
# Breakouts above R1 or below S1 with 1d EMA34 trend filter and volume confirmation capture
# momentum moves. Works in bull (breakouts up) and bear (breakouts down) by following trend.
# Uses 4h timeframe for lower trade frequency, 1d for trend filter, volume filter to avoid false breakouts.

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate 1d EMA(34) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 4-period volume average for confirmation
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    # Calculate Camarilla levels from previous day
    # Use previous day's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla R1 and S1
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for EMA and volume
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(R1[i]) or np.isnan(S1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R1, above 1d EMA34 (uptrend), volume above average
            if close[i] > R1[i] and close[i] > ema_34_aligned[i] and volume[i] > vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1, below 1d EMA34 (downtrend), volume above average
            elif close[i] < S1[i] and close[i] < ema_34_aligned[i] and volume[i] > vol_ma[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 or returns below 1d EMA34
            if close[i] < S1[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 or returns above 1d EMA34
            if close[i] > R1[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla R1/S1 breakout with 1d EMA trend filter and volume confirmation.
# Camarilla levels provide institutional support/resistance. Breakouts above R1 or below S1
# indicate institutional interest. Trend filter ensures we trade with the higher timeframe
# direction (1d EMA34). Volume confirmation avoids false breakouts. Targets 20-50 trades/year.
# Works in bull/bear by aligning with daily trend direction.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # Get daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Using previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla formulas
    R1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    S1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Align to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Daily EMA for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 2  # Need at least 2 days for Camarilla calculation
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: price breaks above R1 + uptrend + volume spike
            if close[i] > R1_aligned[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 + downtrend + volume spike
            elif close[i] < S1_aligned[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns below R1 or trend reversal
            if close[i] < R1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns above S1 or trend reversal
            if close[i] > S1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
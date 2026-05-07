#!/usr/bin/env python3
# 4h_Trend_Following_With_Volume_Filter
# Hypothesis: Combines 4h EMA trend filter with volume spike confirmation to capture institutional momentum moves.
# Uses 12h EMA for multi-timeframe alignment to reduce false signals in both bull and bear markets.
# Target: 20-35 trades/year with strict entry conditions to minimize fee drain.

timeframe = "4h"
name = "4h_Trend_Following_With_Volume_Filter"
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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    
    # Calculate EMA34 on 12h closes
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Get 12h data for EMA200 trend filter
    ema_200_12h = pd.Series(df_12h['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # Volume spike detection: 2x average volume (24-period = 1 day on 4h chart)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 200)  # Ensure we have volume MA and EMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(ema_200_12h_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > EMA34 and EMA200 with volume spike
            if close[i] > ema_34_12h_aligned[i] and close[i] > ema_200_12h_aligned[i] and volume[i] > 2.0 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < EMA34 and EMA200 with volume spike
            elif close[i] < ema_34_12h_aligned[i] and close[i] < ema_200_12h_aligned[i] and volume[i] > 2.0 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below EMA34
            if close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above EMA34
            if close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
name = "4h_WilliamsAlligator_1dTrend_Filter"
timeframe = "4h"
leverage = 1.0

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
    
    # 1d trend filter: EMA34 (Williams Alligator jaw)
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Williams Alligator on 4h: Jaw (EMA13), Teeth (EMA8), Lips (EMA5)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    ema8 = close_series.ewm(span=8, adjust=False, min_periods=8).mean().values
    ema5 = close_series.ewm(span=5, adjust=False, min_periods=5).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # need enough data for 1d EMA34
    
    for i in range(start_idx, n):
        # Skip if 1d trend data not ready
        if np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Alligator aligned upward (Lips > Teeth > Jaw) + price above Jaw + 1d uptrend
            if (ema5[i] > ema8[i] and ema8[i] > ema13[i] and  # Alligator mouth open up
                close[i] > ema13[i] and                       # price above jaw
                close[i] > ema34_1d_aligned[i]):              # 1d uptrend filter
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned downward (Lips < Teeth < Jaw) + price below Jaw + 1d downtrend
            elif (ema5[i] < ema8[i] and ema8[i] < ema13[i] and  # Alligator mouth open down
                  close[i] < ema13[i] and                       # price below jaw
                  close[i] < ema34_1d_aligned[i]):              # 1d downtrend filter
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when Alligator reverses (Lips < Teeth) or price crosses below Jaw
            if (ema5[i] < ema8[i] or close[i] < ema13[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when Alligator reverses (Lips > Teeth) or price crosses above Jaw
            if (ema5[i] > ema8[i] or close[i] > ema13[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
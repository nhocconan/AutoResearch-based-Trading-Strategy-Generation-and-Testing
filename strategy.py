# 12h_1w_HighLow_Filter
# Hypothesis: Uses 1-week high and low as key support/resistance levels on 12h timeframe.
# Long when price crosses above 1w high with 12h above 12-period EMA (uptrend).
# Short when price crosses below 1w low with 12h below 12-period EMA (downtrend).
# Weekly levels act as strong institutional levels; EMA filter avoids counter-trend trades.
# Designed for low trade frequency (12-37/year) to minimize fee drag on 12h chart.

name = "12h_1w_HighLow_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 1W Data for Weekly High/Low and Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's high and low
    ph_1w = high_1w  # previous week high
    pl_1w = low_1w   # previous week low
    
    # 1w EMA12 for trend (using weekly close)
    ema12_1w = pd.Series(close_1w).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # Align weekly data to 12h timeframe
    ph_aligned = align_htf_to_ltf(prices, df_1w, ph_1w)
    pl_aligned = align_htf_to_ltf(prices, df_1w, pl_1w)
    ema12_aligned = align_htf_to_ltf(prices, df_1w, ema12_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers 1w EMA12)
    start_idx = 12
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ph_aligned[i]) or np.isnan(pl_aligned[i]) or 
            np.isnan(ema12_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: close above previous week's high with uptrend (price > weekly EMA)
            if (close[i] > ph_aligned[i] and 
                close[i] > ema12_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: close below previous week's low with downtrend (price < weekly EMA)
            elif (close[i] < pl_aligned[i] and 
                  close[i] < ema12_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below previous week's low (trend invalidation)
            if close[i] < pl_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: close above previous week's high (trend invalidation)
            if close[i] > ph_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals
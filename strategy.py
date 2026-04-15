#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with volume and 12h EMA trend filter
# Combines volatility contraction (BB squeeze) with breakout direction confirmed by volume
# and higher timeframe trend (12h EMA). Works in both bull and bear markets by
# capturing explosive moves after low volatility periods. Target: 50-150 total trades.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Load 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate Bollinger Bands (20, 2.0) on daily
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Bollinger Band Width (squeeze indicator)
    bb_width = (upper_bb - lower_bb) / sma_20
    bb_width = bb_width.values
    
    # Calculate 12h EMA (50-period)
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 4h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb.values)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb.values)
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or
            np.isnan(bb_width_aligned[i]) or np.isnan(ema_50_12h_aligned[i])):
            continue
        
        # Bollinger squeeze condition: BB width below 20-period median (low volatility)
        bb_width_median = np.median(bb_width_aligned[max(0, i-40):i+1])
        is_squeeze = bb_width_aligned[i] < 0.8 * bb_width_median
        
        # Long entry: squeeze breakout above upper BB + volume + 12h EMA uptrend
        if (is_squeeze and close[i] > upper_bb_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            close[i] > ema_50_12h_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: squeeze breakout below lower BB + volume + 12h EMA downtrend
        elif (is_squeeze and close[i] < lower_bb_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              close[i] < ema_50_12h_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse squeeze breakout or volatility expansion (end of squeeze)
        elif position == 1 and close[i] < lower_bb_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > upper_bb_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Bollinger_Squeeze_12hEMA_Volume"
timeframe = "4h"
leverage = 1.0
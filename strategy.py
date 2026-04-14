#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 12h EMA alignment with volume confirmation
# Uses EMA(9) and EMA(21) on 4h and 12h timeframes for trend alignment
# Long when both timeframes show bullish alignment (EMA9 > EMA21)
# Short when both timeframes show bearish alignment (EMA9 < EMA21)
# Volume > 1.5x 20-period EMA confirms momentum
# Target: 20-30 trades/year with trend-following logic
# Stops via EMA crossover in opposite direction

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA(9) and EMA(21)
    close_series = pd.Series(close)
    ema9_4h = close_series.ewm(span=9, adjust=False, min_periods=9).values
    ema21_4h = close_series.ewm(span=21, adjust=False, min_periods=21).values
    
    # Calculate 12h EMA(9) and EMA(21)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema9_12h = close_12h_series.ewm(span=9, adjust=False, min_periods=9).values
    ema21_12h = close_12h_series.ewm(span=21, adjust=False, min_periods=21).values
    
    # Align 12h EMAs to 4h timeframe
    ema9_12h_aligned = align_htf_to_ltf(prices, df_12h, ema9_12h)
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    
    # Volume moving average for confirmation (20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        # Check for NaN values
        if np.isnan(ema9_4h[i]) or np.isnan(ema21_4h[i]) or \
           np.isnan(ema9_12h_aligned[i]) or np.isnan(ema21_12h_aligned[i]) or \
           np.isnan(vol_ma[i]):
            continue
        
        # Volume confirmation (1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Check EMA alignment on both timeframes
        bullish_4h = ema9_4h[i] > ema21_4h[i]
        bearish_4h = ema9_4h[i] < ema21_4h[i]
        bullish_12h = ema9_12h_aligned[i] > ema21_12h_aligned[i]
        bearish_12h = ema9_12h_aligned[i] < ema21_12h_aligned[i]
        
        if position == 0:  # No position - look for trend entries
            # Long when both timeframes show bullish alignment with volume
            if bullish_4h and bullish_12h and volume_confirm:
                position = 1
                signals[i] = position_size
            # Short when both timeframes show bearish alignment with volume
            elif bearish_4h and bearish_12h and volume_confirm:
                position = -1
                signals[i] = -position_size
        elif position == 1:  # Long position - exit on bearish alignment
            # Exit if either timeframe shows bearish alignment
            if bearish_4h or bearish_12h:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit on bullish alignment
            # Exit if either timeframe shows bullish alignment
            if bullish_4h or bullish_12h:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_12h_EMA_Alignment_Volume"
timeframe = "4h"
leverage = 1.0
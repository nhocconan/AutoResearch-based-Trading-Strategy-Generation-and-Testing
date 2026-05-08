#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day ATR breakout with volume confirmation and 1-day EMA trend filter.
# Long when price breaks above daily high + ATR(14) with volume surge and above daily EMA(34).
# Short when price breaks below daily low - ATR(14) with volume surge and below daily EMA(34).
# Uses daily volatility breakout to capture momentum in both trending and ranging markets.
# Designed for low trade frequency (12-30/year) to avoid fee drag. ATR breakouts work in all regimes.

name = "12h_1dATRBreakout_VolumeTrend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR and EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR(14)
    atr_14 = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < 14:
            if i == 0:
                atr_14[i] = tr[0]
            else:
                atr_14[i] = (atr_14[i-1] * i + tr[i]) / (i + 1)
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate 1d EMA(34)
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate breakout levels
    upper_break = high_1d + atr_14
    lower_break = low_1d - atr_14
    
    # Align 1d indicators to 12h timeframe
    upper_break_aligned = align_htf_to_ltf(prices, df_1d, upper_break)
    lower_break_aligned = align_htf_to_ltf(prices, df_1d, lower_break)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: 12h volume spike (2x 20-period EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_break_aligned[i]) or 
            np.isnan(lower_break_aligned[i]) or 
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above daily high + ATR + volume surge + above daily EMA
            if close[i] > upper_break_aligned[i] and vol_spike[i] and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below daily low - ATR + volume surge + below daily EMA
            elif close[i] < lower_break_aligned[i] and vol_spike[i] and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below daily EMA
            if close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above daily EMA
            if close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
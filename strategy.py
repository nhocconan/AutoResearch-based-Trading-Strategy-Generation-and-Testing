#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Uses daily Camarilla pivot levels (R1, S1) and 1-day trend to capture breakouts in both bull and bear markets.
# In bull markets, buy at R1 breakout with bullish trend; in bear markets, sell at S1 breakdown with bearish trend.
# Volume confirmation filters false breakouts. Designed for 4h timeframe with ~30-60 trades/year.

timeframe = "4h"
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # Get daily data for Camarilla pivot points and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla pivot levels
    # R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    range_1d = high_1d - low_1d
    r1 = close_1d + range_1d * 1.1 / 12
    s1 = close_1d - range_1d * 1.1 / 12
    
    # Align pivot points to 4h timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1-day EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 2x average volume (28-period = ~1 day on 4h chart)
    vol_ma = pd.Series(volume).rolling(window=28, min_periods=28).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 28)  # Ensure we have EMA34 and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume, and 1d trend is bullish (close > EMA34)
            if (close[i] > r1_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S1 with volume, and 1d trend is bearish (close < EMA34)
            elif (close[i] < s1_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: price returns below R1 (mean reversion to pivot area)
            if close[i] < r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price returns above S1 (mean reversion to pivot area)
            if close[i] > s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
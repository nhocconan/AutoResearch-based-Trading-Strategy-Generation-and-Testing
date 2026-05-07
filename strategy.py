#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: 4h chart strategy using Camarilla R1/S1 breakouts filtered by daily trend (EMA34) and volume confirmation.
# Long when price breaks above Camarilla R1 with volume > 1.5x average and price > daily EMA34.
# Short when price breaks below Camarilla S1 with volume > 1.5x average and price < daily EMA34.
# Exit on opposite Camarilla level touch (S1 for long, R1 for short).
# Camarilla levels provide institutional support/resistance, EMA34 filters trend direction,
# volume reduces false breakouts. Target: 20-40 trades/year per symbol.

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
    
    # Get daily data for trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Camarilla levels from previous day's OHLC
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = close_1d + (df_1d['high'].values - df_1d['low'].values) * 1.1 / 12.0
    camarilla_s1 = close_1d - (df_1d['high'].values - df_1d['low'].values) * 1.1 / 12.0
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume spike detection: 1.5x average volume (6-period = 1.5 days on 4h chart)
    vol_ma = pd.Series(volume).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 6)  # Ensure we have EMA and volume EMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 with volume, and price > daily EMA34 (uptrend)
            if (high[i] > camarilla_r1_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i] and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 with volume, and price < daily EMA34 (downtrend)
            elif (low[i] < camarilla_s1_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price touches Camarilla S1 (opposite level)
            if low[i] <= camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price touches Camarilla R1 (opposite level)
            if high[i] >= camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
# 12H_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels (R1/S1) from the 1-day timeframe act as strong support/resistance levels.
# Price breaking above R1 with volume confirmation and daily uptrend signals a long entry.
# Price breaking below S1 with volume confirmation and daily downtrend signals a short entry.
# Exits occur when price returns to the daily pivot point (PP). This structure provides clear entry/exit rules
# with low trade frequency, suitable for 12h timeframe. Works in both bull and bear markets by following the daily trend.

name = "12H_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot levels: based on previous day's OHLC
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    # PP = (H+L+C)/3
    range_1d = high_1d - low_1d
    r1_1d = close_1d + range_1d * 1.1 / 12
    s1_1d = close_1d - range_1d * 1.1 / 12
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Align Camarilla levels to 12h timeframe (wait for daily bar to close)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    
    # Daily trend filter: EMA 34 on close
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5 x 24-period average (2 days of 12h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Need enough history for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(pp_1d_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Price breaks above R1 + volume confirmation + daily uptrend
            if close[i] > r1_1d_aligned[i] and volume[i] > vol_threshold[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below S1 + volume confirmation + daily downtrend
            elif close[i] < s1_1d_aligned[i] and volume[i] > vol_threshold[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price returns to or below the daily pivot point (PP)
            if close[i] <= pp_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price returns to or above the daily pivot point (PP)
            if close[i] >= pp_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
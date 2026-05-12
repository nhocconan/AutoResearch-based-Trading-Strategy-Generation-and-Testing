#!/usr/bin/env python3
# 12H_CAMARILLA_R1_S1_BREAKOUT_1W_TREND_FILTER
# Hypothesis: Camarilla pivot levels (R1/S1) from daily act as strong support/resistance.
# In 1-week uptrend (EMA34), go long when price breaks above R1 with volume confirmation.
# In 1-week downtrend (EMA34), go short when price breaks below S1 with volume confirmation.
# Weekly trend filter ensures we only trade with the major trend, reducing whipsaw in both bull and bear markets.
# Target: 15-25 trades/year on 12h timeframe.

name = "12H_CAMARILLA_R1_S1_BREAKOUT_1W_TREND_FILTER"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # EMA34 for weekly trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Camarilla levels from previous day's OHLC
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (using previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Weekly uptrend + price breaks above R1 + volume confirmation
            if (close[i] > ema34_1w_aligned[i] and 
                close[i] > camarilla_r1_aligned[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Weekly downtrend + price breaks below S1 + volume confirmation
            elif (close[i] < ema34_1w_aligned[i] and 
                  close[i] < camarilla_s1_aligned[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or price breaks below S1 (reversal signal)
            if (close[i] <= ema34_1w_aligned[i] or 
                close[i] < camarilla_s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or price breaks above R1 (reversal signal)
            if (close[i] >= ema34_1w_aligned[i] or 
                close[i] > camarilla_r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
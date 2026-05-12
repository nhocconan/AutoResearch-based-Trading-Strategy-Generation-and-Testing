#!/usr/bin/env python3
# 4H_CAMARILLA_R1_S1_BREAKOUT_12H_EMA50_TREND_VOLUME_SPIKE
# Hypothesis: Camarilla R1/S1 levels act as strong intraday support/resistance.
# In 12h uptrend (EMA50), go long when price breaks above R1 with volume spike;
# in 12h downtrend, go short when price breaks below S1 with volume spike.
# Volume spike confirms institutional interest. Works in both bull/bear markets:
# trend filter avoids counter-trend trades, Camarilla breakout captures momentum.
# Target: 20-40 trades/year on 4h timeframe.

name = "4H_CAMARILLA_R1_S1_BREAKOUT_12H_EMA50_TREND_VOLUME_SPIKE"
timeframe = "4h"
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
    
    # Daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla R1 and S1 levels
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # 12h data for trend filter and volume average
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # EMA50 for trend filter
    ema50 = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period) for spike detection
    vol_avg = pd.Series(df_12h['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50)
    vol_avg_aligned = align_htf_to_ltf(prices, df_12h, vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 12h uptrend + price breaks above R1 + volume spike
            if (close[i] > ema50_aligned[i] and 
                close[i] > R1_aligned[i] and 
                volume[i] > 2.0 * vol_avg_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: 12h downtrend + price breaks below S1 + volume spike
            elif (close[i] < ema50_aligned[i] and 
                  close[i] < S1_aligned[i] and 
                  volume[i] > 2.0 * vol_avg_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or price breaks below S1 (invalidates bullish breakout)
            if (close[i] <= ema50_aligned[i] or 
                close[i] < S1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or price breaks above R1 (invalidates bearish breakout)
            if (close[i] >= ema50_aligned[i] or 
                close[i] > R1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
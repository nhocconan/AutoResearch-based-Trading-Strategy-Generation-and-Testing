#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hTrend_Volume
# Hypothesis: In multi-year crypto markets, Camarilla pivot levels (R1/S1) act as strong support/resistance zones.
# Price breaking above R1 in a 12-hour uptrend with volume confirmation indicates bullish momentum.
# Price breaking below S1 in a 12-hour downtrend with volume confirmation indicates bearish momentum.
# Uses 12h EMA50 for trend filter and volume confirmation to filter low-conviction breakouts.
# Works in bull markets (follows uptrends) and bear markets (follows downtrends) via symmetric logic.

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from previous day (using daily data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values  # Shift by 1 to get previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla R1 and S1 levels
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation (20-period MA on 4h chart)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50_12h (50), volume MA (20), and valid Camarilla levels
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_ma[i]) or
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 12h trend filter
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: 12h uptrend + price breaks above R1 + volume
            if uptrend and close[i] > R1_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: 12h downtrend + price breaks below S1 + volume
            elif downtrend and close[i] < S1_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters R1-S1 range
            if not uptrend or (close[i] < R1_aligned[i] and close[i] > S1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price re-enters R1-S1 range
            if not downtrend or (close[i] < R1_aligned[i] and close[i] > S1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
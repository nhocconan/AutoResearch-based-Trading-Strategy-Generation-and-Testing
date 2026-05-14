#!/usr/bin/env python3
# 4h_Camarilla_R2S2_Breakout_12hTrend_Volume
# Hypothesis: Uses Camarilla pivot levels (R2/S2) from 1d timeframe for breakout signals.
# Goes long when price breaks above R2 with volume confirmation and 12h trend filter.
# Goes short when price breaks below S2 with volume confirmation and 12h trend filter.
# Designed to work in both bull and bear markets by aligning with higher timeframe trend.
# Targets 20-50 trades per year on 4h timeframe with position size 0.25.
# Uses 12h EMA(50) for trend filter to avoid counter-trend trades.
# Volume confirmation requires volume > 1.5x 20-period average to confirm breakout strength.
# R2/S2 levels provide less extreme breakouts than R3/S3, reducing false signals while capturing meaningful moves.

name = "4h_Camarilla_R2S2_Breakout_12hTrend_Volume"
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
    
    # Get 1d data for Camarilla pivot levels (using previous day's data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # R2 = Close + (High - Low) * 1.1/4
    # S2 = Close - (High - Low) * 1.1/4
    # Using previous day's data to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate R2 and S2
    r2 = prev_close + (prev_high - prev_low) * 1.1 / 4
    s2 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend direction
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate volume average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Warmup for volume MA and 12h EMA
    
    for i in range(start_idx, n):
        if np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 12h
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: price breaks above R2 with volume confirmation and 12h uptrend
            if close[i] > r2_aligned[i] and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S2 with volume confirmation and 12h downtrend
            elif close[i] < s2_aligned[i] and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below R2 or trend turns down
            if close[i] < r2_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above S2 or trend turns up
            if close[i] > s2_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
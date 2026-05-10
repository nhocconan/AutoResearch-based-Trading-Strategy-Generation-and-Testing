#!/usr/bin/env python3
# 1D_Camarilla_R1_S1_Breakout_1wEMA50_Volume
# Hypothesis: Camarilla pivot R1/S1 breakout with 1-week EMA50 trend filter and volume confirmation.
# Long when price breaks above R1 with volume > 1.5x average and price > 1w EMA50.
# Short when price breaks below S1 with volume > 1.5x average and price < 1w EMA50.
# Exits when price crosses 1w EMA50 in opposite direction.
# Designed for 10-30 trades/year on daily timeframe to avoid overtrading and work in both bull and bear markets.

name = "1D_Camarilla_R1_S1_Breakout_1wEMA50_Volume"
timeframe = "1d"
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
    
    # Calculate Camarilla pivot points from previous day
    pivot = np.full(n, np.nan)
    r1 = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous day's OHLC
        high_prev = high[i-1]
        low_prev = low[i-1]
        close_prev = close[i-1]
        
        pivot[i] = (high_prev + low_prev + close_prev) / 3
        range_prev = high_prev - low_prev
        r1[i] = close_prev + range_prev * 1.1 / 12
        s1[i] = close_prev - range_prev * 1.1 / 12
    
    # Volume average (20)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # 1w EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(pivot[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above R1 with volume confirmation and uptrend
            if close[i] > r1[i] and volume[i] > 1.5 * vol_ma[i] and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below S1 with volume confirmation and downtrend
            elif close[i] < s1[i] and volume[i] > 1.5 * vol_ma[i] and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price crosses below 1w EMA50
            if close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price crosses above 1w EMA50
            if close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
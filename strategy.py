#!/usr/bin/env python3
"""
4h_KAMA_Trend_Direction_12h_TrendFilter
Hypothesis: Use KAMA (Kaufman Adaptive Moving Average) on 4h to determine trend direction, filtered by 12h EMA trend and volume confirmation. Long when KAMA slope turns positive in 12h uptrend with volume above average, short when slope turns negative in 12h downtrend with volume above average. Exit when KAMA slope reverses. Designed for 4h to capture trend changes with low frequency (target 20-50 trades/year).
"""

name = "4h_KAMA_Trend_Direction_12h_TrendFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate KAMA on 4h
    # Efficiency ratio: ER = |close - close[period]| / sum(|diff|) over period
    period = 10
    change = np.abs(np.subtract(close[period:], close[:-period]))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # temporary fix, will compute properly below
    # Recalculate volatility properly
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    # Now compute ER for each point
    er = np.zeros_like(close)
    for i in range(period, len(close)):
        price_change = np.abs(close[i] - close[i-period])
        sum_vol = volatility[i] - volatility[i-period]
        if sum_vol > 0:
            er[i] = price_change / sum_vol
        else:
            er[i] = 0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA slope (direction)
    kama_slope = np.diff(kama, prepend=kama[0])
    
    # Get 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation: 20-period average on 4h
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, period)  # Warmup for 12h EMA34 and KAMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_slope[i]) or np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 12h trend determination
        trend_12h_up = ema_34_12h_aligned[i] > 0  # EMA value itself indicates trend when compared to price, but we need price vs EMA
        # Actually, we need 12h price vs its EMA
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        if np.isnan(close_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        trend_12h_up = close_12h_aligned[i] > ema_34_12h_aligned[i]
        trend_12h_down = close_12h_aligned[i] < ema_34_12h_aligned[i]
        
        if position == 0:
            # Long: KAMA slope turns positive in 12h uptrend with volume confirmation
            if (kama_slope[i] > 0 and kama_slope[i-1] <= 0 and  # slope just turned positive
                trend_12h_up and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: KAMA slope turns negative in 12h downtrend with volume confirmation
            elif (kama_slope[i] < 0 and kama_slope[i-1] >= 0 and  # slope just turned negative
                  trend_12h_down and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA slope turns negative
            if kama_slope[i] < 0 and kama_slope[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA slope turns positive
            if kama_slope[i] > 0 and kama_slope[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
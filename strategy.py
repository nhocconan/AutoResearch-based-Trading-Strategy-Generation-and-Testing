#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Keltner_Breakout_Trend_Volume_WeeklyFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate daily ATR(14) for Keltner channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    
    # Calculate 12-period EMA of close for Keltner basis
    ema12 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema12_aligned = align_htf_to_ltf(prices, df_1d, ema12)
    
    # Keltner Channels: EMA(12) ± 2 * ATR(14)
    keltner_upper = ema12_aligned + 2.0 * atr14_aligned
    keltner_lower = ema12_aligned - 2.0 * atr14_aligned
    
    # Volume spike: current 12h volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(keltner_upper[i]) or 
            np.isnan(keltner_lower[i]) or np.isnan(ema12_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        weekly_trend_up = close[i] > ema20_1w_aligned[i]
        weekly_trend_down = close[i] < ema20_1w_aligned[i]
        upper = keltner_upper[i]
        lower = keltner_lower[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above Keltner upper + weekly uptrend + volume spike
            if (close[i] > upper and 
                weekly_trend_up and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Keltner lower + weekly downtrend + volume spike
            elif (close[i] < lower and 
                  weekly_trend_down and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Keltner lower OR weekly trend turns down
            if (close[i] < lower or not weekly_trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Keltner upper OR weekly trend turns up
            if (close[i] > upper or not weekly_trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
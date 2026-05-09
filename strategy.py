#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Keltner_Breakout_1dTrend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Get 1d data for Keltner Channel (based on previous day's ATR)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate ATR(20) on 1d
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])
    atr20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Keltner Channel: middle = EMA(20), upper/lower = middle ± 2*ATR
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    keltner_upper = ema20_1d + 2 * atr20
    keltner_lower = ema20_1d - 2 * atr20
    
    # Shift to use previous day's values (no look-ahead)
    keltner_upper_prev = np.roll(keltner_upper, 1)
    keltner_lower_prev = np.roll(keltner_lower, 1)
    keltner_upper_prev[0] = np.nan
    keltner_lower_prev[0] = np.nan
    
    # Align Keltner levels to 4h
    keltner_upper_4h = align_htf_to_ltf(prices, df_1d, keltner_upper_prev)
    keltner_lower_4h = align_htf_to_ltf(prices, df_1d, keltner_lower_prev)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(keltner_upper_4h[i]) or np.isnan(keltner_lower_4h[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5 x 20-period average (balanced)
        vol_spike = volume[i] > vol_avg[i] * 1.5
        
        if position == 0:
            # Long: Break above Keltner upper with uptrend and volume spike
            if close[i] > keltner_upper_4h[i] and close[i] > ema50_1d_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below Keltner lower with downtrend and volume spike
            elif close[i] < keltner_lower_4h[i] and close[i] < ema50_1d_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below Keltner lower OR trend turns down
            if close[i] < keltner_lower_4h[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above Keltner upper OR trend turns up
            if close[i] > keltner_upper_4h[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
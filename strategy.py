#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KeltnerBreakout_VolumeTrend_1d"
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
    
    # Get 1d data for Keltner channel and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d Keltner Channel: EMA(20) ± ATR(10)*2
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr10_1d = calculate_atr(high_1d, low_1d, close_1d, 10)
    upper_1d = ema20_1d + (atr10_1d * 2)
    lower_1d = ema20_1d - (atr10_1d * 2)
    
    # Align 1d Keltner to 4h
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    
    # Get 1d trend: EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current 4h volume > 1.3 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema20_1d_aligned[i]) or
            np.isnan(upper_1d_aligned[i]) or
            np.isnan(lower_1d_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema20_val = ema20_1d_aligned[i]
        upper_val = upper_1d_aligned[i]
        lower_val = lower_1d_aligned[i]
        ema50_val = ema50_1d_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: close above upper Keltner + above EMA50 + volume filter
            if close[i] > upper_val and close[i] > ema50_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: close below lower Keltner + below EMA50 + volume filter
            elif close[i] < lower_val and close[i] < ema50_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below EMA20
            if close[i] < ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above EMA20
            if close[i] > ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

def calculate_atr(high, low, close, period):
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_TrueRangeBreakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # first value
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d EMA(50) for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 6h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 6-period range for breakout (6h window)
    range_6h = pd.Series(high - low).rolling(window=6, min_periods=6).max().values
    
    # Volume filter: current 6h volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 50)  # Need enough data for ATR and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(range_6h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        atr = atr_1d_aligned[i]
        trend = ema50_1d_aligned[i]
        rng = range_6h[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: break above 6h high + ATR filter + above trend
            if close[i] > high[i-1] and rng > (atr * 0.5) and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below 6h low + ATR filter + below trend
            elif close[i] < low[i-1] and rng > (atr * 0.5) and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below 6h low or trend reversal
            if close[i] < low[i-1] or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above 6h high or trend reversal
            if close[i] > high[i-1] or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
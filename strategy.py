#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Keltner_Channel_Breakout_1dTrend_Volume"
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
    
    # Get 1d data for trend and Keltner parameters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA for trend
    ema20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate 1d ATR for Keltner channels
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift())
    tr3 = abs(df_1d['low'] - df_1d['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1d EMA for Keltner center
    ema_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Keltner channels: center ± 2 * ATR
    upper_keltner = ema_aligned + 2 * atr_1d_aligned
    lower_keltner = ema_aligned - 2 * atr_1d_aligned
    
    # Volume filter: current 4h volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 20)  # EMA20 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema20_1d_aligned[i]) or
            np.isnan(upper_keltner[i]) or
            np.isnan(lower_keltner[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema20_val = ema20_1d_aligned[i]
        upper = upper_keltner[i]
        lower = lower_keltner[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: close above upper Keltner + above EMA20 trend + volume filter
            if close[i] > upper and close[i] > ema20_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: close below lower Keltner + below EMA20 trend + volume filter
            elif close[i] < lower and close[i] < ema20_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below EMA20 trend
            if close[i] < ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above EMA20 trend
            if close[i] > ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
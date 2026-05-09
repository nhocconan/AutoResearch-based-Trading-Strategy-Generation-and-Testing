#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Keltner_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR, EMA, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA20 for trend
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 1d ATR(10) for Keltner channels
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # 1d Keltner channels
    keltner_upper = ema20_1d + 2.0 * atr10
    keltner_lower = ema20_1d - 2.0 * atr10
    
    # 1d volume average for volume filter
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 12h
    keltner_upper_12h = align_htf_to_ltf(prices, df_1d, keltner_upper)
    keltner_lower_12h = align_htf_to_ltf(prices, df_1d, keltner_lower)
    ema20_12h = align_htf_to_ltf(prices, df_1d, ema20_1d)
    vol_avg_12h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(keltner_upper_12h[i]) or np.isnan(keltner_lower_12h[i]) or 
            np.isnan(ema20_12h[i]) or np.isnan(vol_avg_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper = keltner_upper_12h[i]
        lower = keltner_lower_12h[i]
        trend = ema20_12h[i]
        vol_avg = vol_avg_12h[i]
        vol_ok = volume[i] > vol_avg * 1.5
        
        if position == 0:
            # Long: break above upper Keltner band with volume and above EMA20
            if close[i] > upper and vol_ok and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Keltner band with volume and below EMA20
            elif close[i] < lower and vol_ok and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below lower Keltner band or trend reversal
            if close[i] < lower or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above upper Keltner band or trend reversal
            if close[i] > upper or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
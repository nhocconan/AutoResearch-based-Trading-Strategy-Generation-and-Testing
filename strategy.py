#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Keltner_Channel_Momentum_Breakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Keltner Channel (ATR-based)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1d ATR(10) for Keltner Channel
    tr = np.maximum(np.maximum(df_1d['high'] - df_1d['low'], 
                               np.abs(df_1d['high'] - df_1d['close'].shift(1))),
                      np.abs(df_1d['low'] - df_1d['close'].shift(1)))
    atr10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # 1d EMA(20) for Keltner Channel middle line
    ema20 = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel: Upper = EMA20 + 2*ATR10, Lower = EMA20 - 2*ATR10
    keltner_upper = ema20 + 2 * atr10
    keltner_lower = ema20 - 2 * atr10
    
    # 1w EMA(50) for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all to 4h
    keltner_upper_4h = align_htf_to_ltf(prices, df_1d, keltner_upper)
    keltner_lower_4h = align_htf_to_ltf(prices, df_1d, keltner_lower)
    ema20_4h = align_htf_to_ltf(prices, df_1d, ema20)
    ema50_1w_4h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(keltner_upper_4h[i]) or np.isnan(keltner_lower_4h[i]) or 
            np.isnan(ema20_4h[i]) or np.isnan(ema50_1w_4h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper = keltner_upper_4h[i]
        lower = keltner_lower_4h[i]
        mid = ema20_4h[i]
        trend = ema50_1w_4h[i]
        vol_ok = volume[i] > vol_avg[i] * 1.8
        
        if position == 0:
            # Long: break above upper band with volume and bullish trend
            if close[i] > upper and vol_ok and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with volume and bearish trend
            elif close[i] < lower and vol_ok and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below middle line or trend reversal
            if close[i] < mid or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above middle line or trend reversal
            if close[i] > mid or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
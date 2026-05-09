#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TripleScreen_VolumeBreakout"
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
    
    # Get 1d data for trend and support/resistance
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA200 for long-term trend
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # 1d ATR for volatility
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]), np.abs(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    atr1d = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 1d Bollinger Bands (20, 2)
    sma20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20_1d + 2 * std20_1d
    lower_bb = sma20_1d - 2 * std20_1d
    
    # Align all to 4h
    ema200_1d_4h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    atr1d_4h = align_htf_to_ltf(prices, df_1d, atr1d)
    upper_bb_4h = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_4h = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema200_1d_4h[i]) or np.isnan(atr1d_4h[i]) or 
            np.isnan(upper_bb_4h[i]) or np.isnan(lower_bb_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema200_1d_4h[i]
        atr = atr1d_4h[i]
        upper = upper_bb_4h[i]
        lower = lower_bb_4h[i]
        vol_ok = volume[i] > np.nanmedian(volume[max(0, i-20):i]) * 1.5 if i >= 20 else False
        
        if position == 0:
            # Long: price above EMA200, breaking above upper BB with volume
            if close[i] > trend and close[i] > upper and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below EMA200, breaking below lower BB with volume
            elif close[i] < trend and close[i] < lower and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below EMA200 or below lower BB
            if close[i] < trend or close[i] < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above EMA200 or above upper BB
            if close[i] > trend or close[i] > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
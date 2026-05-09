#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_RSI2_MeanReversion_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA50 for trend
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d volume average
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # RSI(2) on 6h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align to 6h
    ema50_1d_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_avg_1d_6h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1d_6h[i]) or np.isnan(vol_avg_1d_6h[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema50_1d_6h[i]
        vol_avg = vol_avg_1d_6h[i]
        vol_ok = volume[i] > vol_avg * 1.5
        
        if position == 0:
            # Mean reversion: RSI < 10 in uptrend (1d EMA50) for long
            # RSI > 90 in downtrend for short
            if rsi[i] < 10 and close[i] > trend and vol_ok:
                signals[i] = 0.25
                position = 1
            elif rsi[i] > 90 and close[i] < trend and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: RSI > 50 or trend reversal
            if rsi[i] > 50 or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: RSI < 50 or trend reversal
            if rsi[i] < 50 or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
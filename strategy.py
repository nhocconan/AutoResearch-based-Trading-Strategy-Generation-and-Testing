#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_LinearRegressionChannel_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and linear regression channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Get 1d volume average for volume filter
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Linear regression channel on 1d closes (20-period)
    close_1d = df_1d['close'].values
    upper_ch = np.full_like(close_1d, np.nan)
    lower_ch = np.full_like(close_1d, np.nan)
    
    for i in range(19, len(close_1d)):
        y = close_1d[i-19:i+1]
        x = np.arange(20)
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        upper_ch[i] = m * 19 + c + 1.5 * np.std(y - (m * x + c))
        lower_ch[i] = m * 19 + c - 1.5 * np.std(y - (m * x + c))
    
    # 1d trend: EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all to 6h
    upper_ch_6h = align_htf_to_ltf(prices, df_1d, upper_ch)
    lower_ch_6h = align_htf_to_ltf(prices, df_1d, lower_ch)
    ema50_1d_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_avg_1d_6h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(upper_ch_6h[i]) or np.isnan(lower_ch_6h[i]) or 
            np.isnan(ema50_1d_6h[i]) or np.isnan(vol_avg_1d_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper = upper_ch_6h[i]
        lower = lower_ch_6h[i]
        trend = ema50_1d_6h[i]
        vol_avg = vol_avg_1d_6h[i]
        vol_ok = volume[i] > vol_avg * 1.5
        
        if position == 0:
            # Long: break above upper channel with volume and above 1d EMA50
            if close[i] > upper and vol_ok and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Short: break below lower channel with volume and below 1d EMA50
            elif close[i] < lower and vol_ok and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below lower channel or trend reversal
            if close[i] < lower or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above upper channel or trend reversal
            if close[i] > upper or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
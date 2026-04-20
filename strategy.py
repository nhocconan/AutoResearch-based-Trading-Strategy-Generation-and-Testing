#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for pivot levels and volatility
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily True Range and ATR
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_low[0] = high_1d[0] - low_1d[0]
    high_close[0] = np.abs(high_1d[0] - close_1d[0])
    low_close[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate daily volume moving average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate weekly close for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(close_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume_1d[i]
        atr = atr_1d_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        
        # Calculate daily pivot points (using previous day's data)
        if i > 0:
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            prev_close = close_1d[i-1]
            pivot = (prev_high + prev_low + prev_close) / 3.0
            r1 = 2 * pivot - prev_low
            s1 = 2 * pivot - prev_high
        else:
            # Use current day's data for first iteration
            pivot = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0
            r1 = 2 * pivot - low_1d[i]
            s1 = 2 * pivot - high_1d[i]
        
        if position == 0:
            # Long: price crosses above S1 with volume confirmation and weekly uptrend
            if (price > s1 and 
                price <= r1 and  # Avoid overextended moves
                vol > 1.3 * vol_ma and 
                price > ema_20_1w_aligned[i] and
                atr > 0):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below R1 with volume confirmation and weekly downtrend
            elif (price < r1 and 
                  price >= s1 and  # Avoid overextended moves
                  vol > 1.3 * vol_ma and 
                  price < ema_20_1w_aligned[i] and
                  atr > 0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below S1 or volume drops significantly
            if price < s1 or vol < 0.6 * vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above R1 or volume drops significantly
            if price > r1 or vol < 0.6 * vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_S1R1_Breakout_VolumeWeeklyTrend"
timeframe = "12h"
leverage = 1.0
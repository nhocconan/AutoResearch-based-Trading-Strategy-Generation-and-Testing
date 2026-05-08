#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume average (20-period)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate Camarilla levels for 1h (based on previous day's high/low/close)
    # We'll use daily pivot points for Camarilla calculation
    df_1d_ohlc = get_htf_data(prices, '1d')
    if len(df_1d_ohlc) < 2:
        return np.zeros(n)
    
    high_1d = df_1d_ohlc['high'].values
    low_1d = df_1d_ohlc['low'].values
    close_1d = df_1d_ohlc['close'].values
    
    # Calculate daily pivot and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Camarilla levels: R1 = close + (range * 1.1/12), S1 = close - (range * 1.1/12)
    r1 = close_1d + (range_hl * 1.1 / 12)
    s1 = close_1d - (range_hl * 1.1 / 12)
    
    # Align Camarilla levels to 1h
    r1_aligned = align_htf_to_ltf(prices, df_1d_ohlc, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d_ohlc, s1)
    
    # Calculate 1h RSI for entry confirmation
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1, 4h uptrend, volume above average, RSI > 50
            long_cond = (close[i] > r1_aligned[i] and 
                        ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1] and
                        volume[i] > vol_avg_1d_aligned[i] and
                        rsi[i] > 50)
            
            # Short: price breaks below S1, 4h downtrend, volume above average, RSI < 50
            short_cond = (close[i] < s1_aligned[i] and 
                         ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1] and
                         volume[i] > vol_avg_1d_aligned[i] and
                         rsi[i] < 50)
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price crosses below S1 or 4h trend turns down
            if close[i] < s1_aligned[i] or ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price crosses above R1 or 4h trend turns up
            if close[i] > r1_aligned[i] or ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
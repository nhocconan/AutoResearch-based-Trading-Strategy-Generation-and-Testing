#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for 50 EMA and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Get 6h high/low for Donchian channel
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate 6h Donchian(20)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    upper = np.full(len(high_6h), np.nan)
    lower = np.full(len(high_6h), np.nan)
    for i in range(20, len(high_6h)):
        upper[i] = np.max(high_6h[i-20:i])
        lower[i] = np.min(low_6h[i-20:i])
    donch_upper_6h = upper
    donch_lower_6h = lower
    donch_upper_6h_aligned = align_htf_to_ltf(prices, df_6h, donch_upper_6h)
    donch_lower_6h_aligned = align_htf_to_ltf(prices, df_6h, donch_lower_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need EMA(50), ATR(14), Donchian(20)
    start_idx = max(50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(donch_upper_6h_aligned[i]) or np.isnan(donch_lower_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        ema = ema_50_1d_aligned[i]
        atr = atr_1d_aligned[i]
        upper = donch_upper_6h_aligned[i]
        lower = donch_lower_6h_aligned[i]
        
        # Trend filter: price above/below EMA
        uptrend = close[i] > ema
        downtrend = close[i] < ema
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr > 0
        
        # Entry conditions: breakout with trend alignment
        if position == 0:
            # Long: break above upper band + uptrend
            if close[i] > upper and uptrend and vol_filter:
                signals[i] = size
                position = 1
            # Short: break below lower band + downtrend
            elif close[i] < lower and downtrend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below EMA or breakdown below lower band
            if close[i] < ema or close[i] < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above EMA or breakout above upper band
            if close[i] > ema or close[i] > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_1dEMA50_TrendFilter"
timeframe = "6h"
leverage = 1.0
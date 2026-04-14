#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels (daily)
    if len(high_1d) < 20:
        return np.zeros(n)
    
    # Upper band: highest high over last 20 periods
    upper_20 = np.full_like(high_1d, np.nan)
    for i in range(19, len(high_1d)):
        upper_20[i] = np.max(high_1d[i-19:i+1])
    
    # Lower band: lowest low over last 20 periods
    lower_20 = np.full_like(low_1d, np.nan)
    for i in range(19, len(low_1d)):
        lower_20[i] = np.min(low_1d[i-19:i+1])
    
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Calculate 10-period ATR for volatility filter (daily)
    if len(high_1d) < 14:
        return np.zeros(n)
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr10 = np.full_like(close_1d, np.nan)
    if len(tr) >= 10:
        atr10[9] = np.mean(tr[1:11])
        for i in range(10, len(tr)):
            atr10[i] = (atr10[i-1] * 9 + tr[i]) / 10
    
    atr10_aligned = align_htf_to_ltf(prices, df_1d, atr10)
    
    # Calculate 50-day EMA for trend filter (daily)
    if len(close_1d) < 50:
        return np.zeros(n)
    
    ema50 = np.full_like(close_1d, np.nan)
    ema50[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema50[i] = close_1d[i] * 0.0392 + ema50[i-1] * 0.9608  # 2/(50+1)
    
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(atr10_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: require ATR > 0.5% of price
        if atr10_aligned[i] / close[i] < 0.005:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian + price above EMA50
            if (close[i] > upper_20_aligned[i] and
                close[i] > ema50_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below lower Donchian + price below EMA50
            elif (close[i] < lower_20_aligned[i] and
                  close[i] < ema50_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price breaks below lower Donchian OR price below EMA50
            if (close[i] < lower_20_aligned[i] or 
                close[i] < ema50_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price breaks above upper Donchian OR price above EMA50
            if (close[i] > upper_20_aligned[i] or 
                close[i] > ema50_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Donchian20_EMA50_VolFilter"
timeframe = "4h"
leverage = 1.0
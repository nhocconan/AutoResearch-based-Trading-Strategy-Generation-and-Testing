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
    
    # Load 1-day data (HTF) once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR for volatility filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    atr_4h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate daily EMA200 for trend filter (1d)
    ema200_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 200:
        ema200_1d[199] = np.mean(close_1d[:200])
        for i in range(200, len(df_1d)):
            ema200_1d[i] = (close_1d[i] * 2 + ema200_1d[i-1] * 198) / 200
    
    ema200_4h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 4-hour Donchian channels (20-period) for entry signals
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(200, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_4h[i]) or
            np.isnan(ema200_4h[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.4% of price)
        if atr_4h[i] < 0.004 * close[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above 4h Donchian high with volatility filter and above daily EMA200
            if close[i] > donch_high[i] and close[i] > ema200_4h[i]:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 4h Donchian low with volatility filter and below daily EMA200
            elif close[i] < donch_low[i] and close[i] < ema200_4h[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below 4h Donchian low OR below daily EMA200
            if close[i] < donch_low[i] or close[i] < ema200_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above 4h Donchian high OR above daily EMA200
            if close[i] > donch_high[i] or close[i] > ema200_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Donchian_EMA200_Volatility"
timeframe = "4h"
leverage = 1.0
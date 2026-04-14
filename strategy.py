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
    
    # Load weekly data (HTF) once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly ATR (14-period) for volatility filter
    tr = np.zeros(len(df_1w))
    tr[0] = high_1w[0] - low_1w[0]
    for i in range(1, len(df_1w)):
        tr[i] = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
    
    atr_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 14:
        atr_1w[13] = np.mean(tr[:14])
        for i in range(14, len(df_1w)):
            atr_1w[i] = (atr_1w[i-1] * 13 + tr[i]) / 14
    
    atr_1d = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate weekly 50-period EMA for trend filter
    ema50_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 50:
        ema50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(df_1w)):
            ema50_1w[i] = (close_1w[i] * 2 + ema50_1w[i-1] * 48) / 50
    
    ema50_1d = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate daily Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_1d[i]) or
            np.isnan(ema50_1d[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_1d[i] < 0.005 * close[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above daily Donchian high with above weekly EMA50
            if close[i] > donch_high[i] and close[i] > ema50_1d[i]:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below daily Donchian low with below weekly EMA50
            elif close[i] < donch_low[i] and close[i] < ema50_1d[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below daily Donchian low OR below weekly EMA50
            if close[i] < donch_low[i] or close[i] < ema50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above daily Donchian high OR above weekly EMA50
            if close[i] > donch_high[i] or close[i] > ema50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian20_EMA50_Trend"
timeframe = "1d"
leverage = 1.0
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
    
    # Hour of day filter (08:00-20:00 UTC) - computed once before loop
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Load daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily ATR (14-period)
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
    
    atr_1h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate daily 20-period EMA
    ema20_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 20:
        ema20_1d[19] = np.mean(close_1d[:20])
        for i in range(20, len(df_1d)):
            ema20_1d[i] = (close_1d[i] * 2 + ema20_1d[i-1] * 18) / 20
    
    ema20_1h = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate 1-hour Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_1h[i]) or
            np.isnan(ema20_1h[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_1h[i] < 0.005 * close[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above 1h Donchian high with volatility filter and above daily EMA20
            if close[i] > donch_high[i] and close[i] > ema20_1h[i]:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 1h Donchian low with volatility filter and below daily EMA20
            elif close[i] < donch_low[i] and close[i] < ema20_1h[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below 1h Donchian low OR below daily EMA20
            if close[i] < donch_low[i] or close[i] < ema20_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above 1h Donchian high OR above daily EMA20
            if close[i] > donch_high[i] or close[i] > ema20_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_1d_Donchian_EMA20_Volatility_Filter"
timeframe = "1h"
leverage = 1.0
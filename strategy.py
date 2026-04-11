#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 4h ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h Camarilla pivot levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    pivot_4h = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    
    # Resistance levels (tighter levels for fewer trades)
    h1_4h = close_4h + 1.1 * range_4h / 6
    h2_4h = close_4h + 1.1 * range_4h / 4
    h3_4h = close_4h + 1.1 * range_4h / 2
    
    # Support levels
    l1_4h = close_4h - 1.1 * range_4h / 6
    l2_4h = close_4h - 1.1 * range_4h / 4
    l3_4h = close_4h - 1.1 * range_4h / 2
    
    # Calculate 1d trend filter (EMA cross)
    ema_fast_1d = pd.Series(close_4h).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_slow_1d = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    trend_1d = ema_fast_1d > ema_slow_1d  # True for uptrend
    
    # Align 4h levels and 1d trend to 1h
    pivot_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    h1_aligned = align_htf_to_ltf(prices, df_4h, h1_4h)
    h2_aligned = align_htf_to_ltf(prices, df_4h, h2_4h)
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3_4h)
    l1_aligned = align_htf_to_ltf(prices, df_4h, l1_4h)
    l2_aligned = align_htf_to_ltf(prices, df_4h, l2_4h)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3_4h)
    trend_aligned = align_htf_to_ltf(prices, df_4h, trend_1d.astype(float))
    
    # Volume confirmation: current volume > 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Session filter: 8-20 UTC
    hours = prices.index.hour
    
    # Start from index 30 to ensure sufficient data
    for i in range(30, n):
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Skip if any required data is invalid
        if (np.isnan(pivot_aligned[i]) or np.isnan(h1_aligned[i]) or np.isnan(l1_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(vol_avg_20[i]) or np.isnan(trend_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_avg_20[i]
        
        # Price levels for current bar
        h1 = h1_aligned[i]
        h2 = h2_aligned[i]
        h3 = h3_aligned[i]
        l1 = l1_aligned[i]
        l2 = l2_aligned[i]
        l3 = l3_aligned[i]
        pivot = pivot_aligned[i]
        trend_up = trend_aligned[i] > 0.5
        
        # Long conditions: break above resistance with volume and uptrend
        long_signal = vol_confirm and trend_up and (
            close[i] > h1 or  # break H1
            close[i] > h2 or  # break H2
            close[i] > h3     # break H3
        )
        
        # Short conditions: break below support with volume and downtrend
        short_signal = vol_confirm and (not trend_up) and (
            close[i] < l1 or  # break L1
            close[i] < l2 or  # break L2
            close[i] < l3     # break L3
        )
        
        # Exit conditions: price returns to pivot
        long_exit = close[i] < pivot
        short_exit = close[i] > pivot
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals
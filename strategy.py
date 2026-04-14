#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h time-based session filter: 08-20 UTC
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    # 1h Donchian channels (20-period) - use previous bar's high/low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # 1h average volume (20-period) - previous bar
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    # 1h EMA200 trend filter
    ema_200_1h = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # 4h EMA50 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 4h EMA200 for trend filter (HTF)
    ema_200_4h = pd.Series(df_4h['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d EMA200 for trend filter (HTF)
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    start = max(20, 200, 14)
    for i in range(start, n):
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(avg_vol[i]) or np.isnan(ema_200_1h[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: breakout above upper band + volume confirmation + price above EMA200_1h + bullish higher TF alignment
            if (price > upper[i] and vol > 2.0 * avg_vol[i] and 
                price > ema_200_1h[i] and 
                ema_50_4h_aligned[i] > ema_200_4h_aligned[i] and
                ema_50_1d_aligned[i] > ema_200_1d_aligned[i] and
                session_mask[i]):  # Only trade during active session
                position = 1
                signals[i] = position_size
            # Short: breakout below lower band + volume confirmation + price below EMA200_1h + bearish higher TF alignment
            elif (price < lower[i] and vol > 2.0 * avg_vol[i] and 
                  price < ema_200_1h[i] and
                  ema_50_4h_aligned[i] < ema_200_4h_aligned[i] and
                  ema_50_1d_aligned[i] < ema_200_1d_aligned[i] and
                  session_mask[i]):  # Only trade during active session
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below lower band OR below EMA200_1h
            if (price < lower[i] or price < ema_200_1h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above upper band OR above EMA200_1h
            if (price > upper[i] or price > ema_200_1h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_1d_Donchian_Volume_EMA200Trend_MultiTF_Filter"
timeframe = "1h"
leverage = 1.0
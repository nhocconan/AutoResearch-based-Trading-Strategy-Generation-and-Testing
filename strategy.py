#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h price - for primary timeframe calculations
    high_12h = pd.Series(high)
    low_12h = pd.Series(low)
    close_12h = pd.Series(close)
    
    # 12h Donchian channels (20-period) - use previous bar's high/low
    upper = high_12h.rolling(window=20, min_periods=20).max().shift(1).values
    lower = low_12h.rolling(window=20, min_periods=20).min().shift(1).values
    
    # 12h average volume (20-period) - previous bar
    avg_vol = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    # Daily EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 1h ATR (14) for volatility filter - using 1h data
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 14:
        return np.zeros(n)
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    tr1 = np.maximum(high_1h[1:] - low_1h[1:], np.abs(high_1h[1:] - close_1h[:-1]))
    tr2 = np.abs(low_1h[1:] - close_1h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, tr2)])
    atr_1h = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    atr_1h_aligned = align_htf_to_ltf(prices, df_1h, atr_1h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    start = max(20, 200, 14)
    for i in range(start, n):
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(avg_vol[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(atr_1h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        atr = atr_1h_aligned[i]
        
        if position == 0:
            # Long: breakout above upper band + volume confirmation + price above daily EMA200
            if (price > upper[i] and vol > 2.0 * avg_vol[i] and price > ema_200_1d_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: breakout below lower band + volume confirmation + price below daily EMA200
            elif (price < lower[i] and vol > 2.0 * avg_vol[i] and price < ema_200_1d_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below lower band OR below daily EMA200 OR trailing stop
            if price < lower[i] or price < ema_200_1d_aligned[i] or price < (high[i] - 2.0 * atr):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above upper band OR above daily EMA200 OR trailing stop
            if price > upper[i] or price > ema_200_1d_aligned[i] or price > (low[i] + 2.0 * atr):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Donchian_Volume_EMA200_ATR"
timeframe = "12h"
leverage = 1.0
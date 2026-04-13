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
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ATR for stop-loss
    high_low = df_1d['high'] - df_1d['low']
    high_close = np.abs(df_1d['high'] - np.roll(df_1d['close'], 1))
    low_close = np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))
    tr_1d = np.maximum(high_low, np.maximum(high_close, low_close))
    tr_1d[0] = high_low[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d Donchian channels (20-period)
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    
    # 1d average volume (20-period)
    vol_1d = df_1d['volume']
    avg_vol_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().shift(1).values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # 1d EMA200 trend filter
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    start = max(20, 200, 14)
    for i in range(start, n):
        if (np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or 
            np.isnan(avg_vol_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: breakout above upper band + volume confirmation + price above EMA200
            if (price > upper_1d_aligned[i] and vol > 2.0 * avg_vol_1d_aligned[i] and price > ema_200_1d_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: breakout below lower band + volume confirmation + price below EMA200
            elif (price < lower_1d_aligned[i] and vol > 2.0 * avg_vol_1d_aligned[i] and price < ema_200_1d_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below lower band OR below EMA200 OR stop-loss hit
            if (price < lower_1d_aligned[i] or price < ema_200_1d_aligned[i] or 
                price < (entry_price := entry_price_long) - 2.0 * atr_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above upper band OR above EMA200 OR stop-loss hit
            if (price > upper_1d_aligned[i] or price > ema_200_1d_aligned[i] or 
                price > (entry_price := entry_price_short) + 2.0 * atr_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        
        # Track entry price for stop-loss calculation
        if position != 0 and signals[i] != 0 and (i == start or signals[i-1] == 0):
            if position == 1:
                entry_price_long = close[i]
            else:
                entry_price_short = close[i]
    
    return signals

name = "12h_1d_Donchian_Volume_EMA200Trend_ATR"
timeframe = "12h"
leverage = 1.0
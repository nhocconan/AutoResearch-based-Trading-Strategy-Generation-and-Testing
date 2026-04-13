#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 300:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period) - use previous bar's high/low
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    high_series_4h = pd.Series(high_4h)
    low_series_4h = pd.Series(low_4h)
    upper_4h = high_series_4h.rolling(window=20, min_periods=20).max().shift(1).values
    lower_4h = low_series_4h.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align 4h levels to 1h timeframe
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # 1d average volume (20-period) - previous bar
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_series_1d = pd.Series(vol_1d)
    avg_vol_1d = vol_series_1d.rolling(window=20, min_periods=20).mean().shift(1).values
    
    # Align 1d average volume to 1h timeframe
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # 1d EMA200 trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 1h ATR (14-period) for stop-loss
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().shift(1).values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20
    
    start = max(20, 200, 14)
    for i in range(start, n):
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or 
            np.isnan(avg_vol_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: breakout above 4h upper band + volume confirmation + price above 1d EMA200
            if (price > upper_4h_aligned[i] and vol > 2.0 * avg_vol_1d_aligned[i] and price > ema_200_1d_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: breakout below 4h lower band + volume confirmation + price below 1d EMA200
            elif (price < lower_4h_aligned[i] and vol > 2.0 * avg_vol_1d_aligned[i] and price < ema_200_1d_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below 4h lower band OR below 1d EMA200 OR stop-loss hit
            if (price < lower_4h_aligned[i] or price < ema_200_1d_aligned[i] or 
                price < entry_price_long - 2.0 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above 4h upper band OR above 1d EMA200 OR stop-loss hit
            if (price > upper_4h_aligned[i] or price > ema_200_1d_aligned[i] or 
                price > entry_price_short + 2.0 * atr[i]):
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

name = "1h_4h_1d_Donchian_Volume_EMA200Trend_SSR"
timeframe = "1h"
leverage = 1.0
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
    
    # 4h Donchian channels (20-period) - use previous bar's high/low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # 4h average volume (20-period) - previous bar
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    # 4h ATR (14-period) for stop-loss
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().shift(1).values
    
    # 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 12h EMA200 for trend filter (HTF)
    ema_200_12h = pd.Series(df_12h['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    start = max(20, 200, 14)
    for i in range(start, n):
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(avg_vol[i]) or np.isnan(atr[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_200_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: breakout above upper band + volume confirmation + price above EMA200_12h (bullish)
            if (price > upper[i] and vol > 2.0 * avg_vol[i] and 
                price > ema_200_12h_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: breakout below lower band + volume confirmation + price below EMA200_12h (bearish)
            elif (price < lower[i] and vol > 2.0 * avg_vol[i] and 
                  price < ema_200_12h_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below lower band OR stop-loss hit
            if (price < lower[i] or 
                price < ema_50_12h_aligned[i] - 2.0 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above upper band OR stop-loss hit
            if (price > upper[i] or 
                price > ema_50_12h_aligned[i] + 2.0 * atr[i]):
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

name = "4h_12h_Donchian_Volume_EMA200Trend_EMA50Filter"
timeframe = "4h"
leverage = 1.0
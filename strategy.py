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
    
    # 1d OHLC from daily data (for Camarilla pivot calculation)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d (using previous day's data)
    # Resistance levels
    R4 = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    R3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    R2 = close_1d + ((high_1d - low_1d) * 1.1 / 6)
    R1 = close_1d + ((high_1d - low_1d) * 1.1 / 12)
    # Support levels
    S1 = close_1d - ((high_1d - low_1d) * 1.1 / 12)
    S2 = close_1d - ((high_1d - low_1d) * 1.1 / 6)
    S3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    S4 = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe (using previous day's values)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # 1d average volume (20-period) - previous bar
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    # 1d EMA200 trend filter
    ema_200_1d = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # 1d ATR (14-period) for stop-loss
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]  # first value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    start = max(20, 200, 14)
    for i in range(start, n):
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(avg_vol[i]) or np.isnan(ema_200_1d[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price touches S1 level + volume confirmation + price above EMA200
            if (price <= S1_aligned[i] * 1.001 and price >= S1_aligned[i] * 0.999 and 
                vol > 2.0 * avg_vol[i] and price > ema_200_1d[i]):
                position = 1
                signals[i] = position_size
            # Short: price touches R1 level + volume confirmation + price below EMA200
            elif (price >= R1_aligned[i] * 0.999 and price <= R1_aligned[i] * 1.001 and 
                  vol > 2.0 * avg_vol[i] and price < ema_200_1d[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price touches R1 level OR below EMA200 OR stop-loss hit
            if (price >= R1_aligned[i] * 0.999 and price <= R1_aligned[i] * 1.001) or \
               price < ema_200_1d[i] or \
               price < (entry_price := entry_price_long) - 2.0 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price touches S1 level OR above EMA200 OR stop-loss hit
            if (price <= S1_aligned[i] * 1.001 and price >= S1_aligned[i] * 0.999) or \
               price > ema_200_1d[i] or \
               price > (entry_price := entry_price_short) + 2.0 * atr[i]:
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

name = "4h_1d_Camarilla_Touch_Volume_EMA200Trend_ATR"
timeframe = "4h"
leverage = 1.0
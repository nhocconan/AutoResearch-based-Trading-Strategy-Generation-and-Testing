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
    
    # 1d high/low for Camarilla pivot calculation
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # Calculate 1d Camarilla pivot levels (using previous day's data)
    # Pivot = (H + L + C) / 3
    pivot = (high_series + low_series + close_series) / 3
    range_hl = high_series - low_series
    
    # Resistance levels
    r1 = pivot + (range_hl * 1.1 / 12)
    r2 = pivot + (range_hl * 1.1 / 6)
    r3 = pivot + (range_hl * 1.1 / 4)
    r4 = pivot + (range_hl * 1.1 / 2)
    
    # Support levels
    s1 = pivot - (range_hl * 1.1 / 12)
    s2 = pivot - (range_hl * 1.1 / 6)
    s3 = pivot - (range_hl * 1.1 / 4)
    s4 = pivot - (range_hl * 1.1 / 2)
    
    # Use previous day's levels to avoid look-ahead
    r1_prev = r1.shift(1).values
    r2_prev = r2.shift(1).values
    s1_prev = s1.shift(1).values
    s2_prev = s2.shift(1).values
    
    # 1d average volume (20-period) - previous bar
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    # 1d EMA200 trend filter
    ema_200_1d = close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # 1d ATR (14-period) for stop-loss
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    start = max(20, 200, 14)
    for i in range(start, n):
        if (np.isnan(r1_prev[i]) or np.isnan(s1_prev[i]) or 
            np.isnan(avg_vol[i]) or np.isnan(ema_200_1d[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price touches S1 support + volume reversal + price above EMA200
            if (price <= s1_prev[i] and vol > 1.5 * avg_vol[i] and price > ema_200_1d[i]):
                position = 1
                signals[i] = position_size
            # Short: price touches R1 resistance + volume reversal + price below EMA200
            elif (price >= r1_prev[i] and vol > 1.5 * avg_vol[i] and price < ema_200_1d[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches S2 support OR below EMA200 OR stop-loss hit
            if (price <= s2_prev[i] or price < ema_200_1d[i] or 
                price < entry_price_long - 2.0 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches R2 resistance OR above EMA200 OR stop-loss hit
            if (price >= r2_prev[i] or price > ema_200_1d[i] or 
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

name = "4h_1d_Camarilla_Pivot_Volume_EMA200Trend_ATR"
timeframe = "4h"
leverage = 1.0
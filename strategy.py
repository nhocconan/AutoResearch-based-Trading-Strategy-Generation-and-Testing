#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly pivot filter and volume confirmation
# Long: Price breaks above 6h Donchian upper + price above weekly pivot (bullish bias) + volume spike
# Short: Price breaks below 6h Donchian lower + price below weekly pivot (bearish bias) + volume spike
# Weekly pivot provides regime filter to avoid counter-trend trades in strong trends
# Volume confirmation reduces false breakouts
# Target: 50-150 trades over 4 years (12-37/year) with size 0.25

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Donchian channels (20-period) - use previous bar's high/low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # 6h average volume (20-period) - previous bar
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    # Weekly pivot levels (using 1w data as HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard floor trader's method)
    # Pivot = (H + L + C)/3, R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    r2 = pivot + (high_1w - low_1w)
    s2 = pivot - (high_1w - low_1w)
    r3 = high_1w + 2 * (pivot - low_1w)
    s3 = low_1w - 2 * (high_1w - pivot)
    
    # Use weekly pivot and R3/S3 as bias filters
    # Bullish bias: price above weekly pivot
    # Bearish bias: price below weekly pivot
    pivot_val = pivot
    bullish_bias = pivot_val
    bearish_bias = pivot_val
    
    pivot_aligned = align_htf_to_ltf(prices, df_1w, bullish_bias)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # ATR for volatility filtering and stop-loss
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    start = max(20, 20)
    for i in range(start, n):
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(avg_vol[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: breakout above Donchian upper + bullish bias (price above weekly pivot) + volume spike
            if (price > upper[i] and price > pivot_aligned[i] and vol > 2.0 * avg_vol[i]):
                position = 1
                signals[i] = position_size
            # Short: breakout below Donchian lower + bearish bias (price below weekly pivot) + volume spike
            elif (price < lower[i] and price < pivot_aligned[i] and vol > 2.0 * avg_vol[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit conditions: price closes below weekly pivot OR Donchian lower OR stop-loss
            if (price < pivot_aligned[i] or price < lower[i] or 
                price < (entry_price := entry_price_long) - 2.5 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit conditions: price closes above weekly pivot OR Donchian upper OR stop-loss
            if (price > pivot_aligned[i] or price > upper[i] or 
                price > (entry_price := entry_price_short) + 2.5 * atr[i]):
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

name = "6h_1w_Donchian_Pivot_Volume_Filter"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with weekly pivot direction + volume confirmation
# Weekly pivot (from prior week) defines bias, price action at 6h breakouts with volume confirm entries
# Works in bull/bear because pivot provides structural support/resistance and volume confirms conviction
# Target: 15-30 trades/year (60-120 over 4 years) to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot calculation (prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Pivot point and support/resistance levels
    pw = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pw - weekly_low
    s1 = 2 * pw - weekly_high
    r2 = pw + (weekly_high - weekly_low)
    s2 = pw - (weekly_high - weekly_low)
    
    # Align weekly levels to 6h timeframe (using prior week's values)
    pw_aligned = align_htf_to_ltf(prices, df_1w, pw)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # 6h ATR for volatility filter and stop-loss
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().shift(1).values
    
    # 6h average volume (20-period) for confirmation
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    start = max(20, 14)  # warmup period
    for i in range(start, n):
        if (np.isnan(pw_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(atr[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: breakout above R1 with volume confirmation
            if (price > r1_aligned[i] and vol > 1.5 * avg_vol[i]):
                position = 1
                signals[i] = position_size
            # Short: breakdown below S1 with volume confirmation
            elif (price < s1_aligned[i] and vol > 1.5 * avg_vol[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below pivot OR stop-loss hit
            if (price < pw_aligned[i] or 
                price < entry_price_long - 2.0 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above pivot OR stop-loss hit
            if (price > pw_aligned[i] or 
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

name = "6h_1w_Pivot_Volume_Breakout"
timeframe = "6h"
leverage = 1.0
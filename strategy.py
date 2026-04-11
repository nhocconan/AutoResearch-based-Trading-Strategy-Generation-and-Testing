#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_volume_v2
# Strategy: 4h Camarilla pivot breakout with volume confirmation and 1d trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla levels (H4/L4, H3/L3) act as strong support/resistance. 
# Breakouts with volume confirmation in the direction of the 1d trend yield high-probability trades.
# Works in bull/bear: In uptrends, buy L3/L4 breaks; in downtrends, sell H3/H4 breaks.
# Low frequency (~20-40/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous 1d candle
    # Typical Price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Range = H - L
    range_1d = df_1d['high'] - df_1d['low']
    # Camarilla levels
    # H4 = Close + 1.5 * Range * 1.1
    # L4 = Close - 1.5 * Range * 1.1
    # H3 = Close + 1.25 * Range * 1.1
    # L3 = Close - 1.25 * Range * 1.1
    h4 = df_1d['close'] + 1.5 * range_1d * 1.1
    l4 = df_1d['close'] - 1.5 * range_1d * 1.1
    h3 = df_1d['close'] + 1.25 * range_1d * 1.1
    l3 = df_1d['close'] - 1.25 * range_1d * 1.1
    
    # Align Camarilla levels to 4h (they are constant throughout the day)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4.values)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4.values)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3.values)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3.values)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.8 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: Camarilla breakout + volume + trend alignment
        if (close[i] > h3_aligned[i] and vol_confirm[i] and uptrend and position != 1):
            # Break above H3 in uptrend -> long
            position = 1
            signals[i] = 0.25
        elif (close[i] < l3_aligned[i] and vol_confirm[i] and downtrend and position != -1):
            # Break below L3 in downtrend -> short
            position = -1
            signals[i] = -0.25
        # Exit: price returns to opposite Camarilla level or trend change
        elif position == 1 and (close[i] < l3_aligned[i] or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > h3_aligned[i] or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
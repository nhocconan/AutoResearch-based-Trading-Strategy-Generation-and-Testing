#!/usr/bin/env python3
"""
6h_12h_Trend_1d_Confluence_Breakout
Hypothesis: On 6h timeframe, take long when price breaks above the 12h EMA20 with
1d EMA50 uptrend and volume confirmation; take short when price breaks below
12h EMA20 with 1d EMA50 downtrend and volume confirmation. Uses EMA trend
alignment across timeframes to filter noise and capture sustained moves.
Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in
downtrend) markets with tight entry conditions targeting 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h EMA20 for trend and breakout level
    df_12h = get_htf_data(prices, '12h')
    ema_12h_20 = pd.Series(df_12h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_12h_20_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_20)
    
    # 1d EMA50 for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_1d_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_12h_20_aligned[i]) or np.isnan(ema_1d_50_aligned[i]) or
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_12h = ema_12h_20_aligned[i]
        ema_1d = ema_1d_50_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above 12h EMA20 with 1d uptrend and volume
            if price > ema_12h and ema_1d > ema_12h and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h EMA20 with 1d downtrend and volume
            elif price < ema_12h and ema_1d < ema_12h and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns below 12h EMA20 or 1d trend turns down
            if price < ema_12h or ema_1d < ema_12h:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns above 12h EMA20 or 1d trend turns up
            if price > ema_12h or ema_1d > ema_12h:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_12h_Trend_1d_Confluence_Breakout"
timeframe = "6h"
leverage = 1.0
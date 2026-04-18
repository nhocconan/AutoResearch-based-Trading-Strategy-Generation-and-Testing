# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
12h_PriceChannel_Breakout_Volume_Trend
Hypothesis: 12-hour price channel breakouts (Donchian 20) with volume confirmation
and daily EMA trend filter capture institutional move initiation. Works in bull/bear
by following directional momentum. Target: 20-40 trades/year (80-160 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 20-period Donchian channels on 12h data
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume filter: >1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (1.8 * vol_ma)
    
    # 1-day EMA trend filter (34-period)
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_12h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(lookback, 30)  # Warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(volume_filter[i]) or np.isnan(ema_1d_12h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_val = upper[i]
        lower_val = lower[i]
        vol_ok = volume_filter[i]
        ema_trend = ema_1d_12h[i]
        
        if position == 0:
            # Long: break above upper band with volume in uptrend
            if price > upper_val and vol_ok and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with volume in downtrend
            elif price < lower_val and vol_ok and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to lower band or trend reverses
            if price < lower_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to upper band or trend reverses
            if price > upper_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_PriceChannel_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0
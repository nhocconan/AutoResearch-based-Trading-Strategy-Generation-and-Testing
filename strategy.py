#!/usr/bin/env python3
# 12h_1d_volume_confirm_trend_v1
# Strategy: 12h price closes above/below 1d EMA(50) with volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Trend-following with EMA(50) filter on daily timeframe. Volume > 1.5x average confirms institutional interest.
# Works in bull by catching uptrends, in bear by avoiding counter-trend trades and capturing rebounds from strong support.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_volume_confirm_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(ema_1d_aligned[i]) or np.isnan(vol_avg[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        # Entry conditions: price relative to daily EMA with volume confirmation
        long_entry = (close[i] > ema_1d_aligned[i]) and vol_spike[i]
        short_entry = (close[i] < ema_1d_aligned[i]) and vol_spike[i]
        
        # Exit conditions: price crosses back through EMA
        exit_long = position == 1 and close[i] < ema_1d_aligned[i]
        exit_short = position == -1 and close[i] > ema_1d_aligned[i]
        
        # Trading logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
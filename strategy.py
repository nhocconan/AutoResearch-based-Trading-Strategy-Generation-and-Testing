# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
1h_Structure_Trend_Filter
Long when: 4h price above SMA200 + 1h price above EMA50 + volume > 1.5x 20-period average.
Short when: 4h price below SMA200 + 1h price below EMA50 + volume > 1.5x 20-period average.
Exit when price crosses back EMA50.
Position size: 0.20. Target: 15-37 trades/year.
Uses 4h for long-term trend (SMA200), 1h for entry/exit (EMA50), volume for momentum.
Avoids overtrading by requiring multi-factor confluence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h SMA200 for trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    sma200_4h = pd.Series(close_4h).rolling(window=200, min_periods=200).mean().values
    sma200_4h_aligned = align_htf_to_ltf(prices, df_4h, sma200_4h)
    
    # 1h EMA50 for entry/exit
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1h volume filter
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(50, n):  # warmup for EMA50
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(sma200_4h_aligned[i]) or np.isnan(ema50[i]) or np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Current price vs 4h SMA200 (trend)
        # Since we don't have 4h close at 1h bar, we use price vs aligned SMA200 as proxy
        price_above_4h_sma200 = close[i] > sma200_4h_aligned[i]
        price_below_4h_sma200 = close[i] < sma200_4h_aligned[i]
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * vol_ma20[i])
        
        if position == 0:
            # Long: uptrend + price above EMA50 + volume
            if price_above_4h_sma200 and close[i] > ema50[i] and volume_filter:
                signals[i] = 0.20
                position = 1
            # Short: downtrend + price below EMA50 + volume
            elif price_below_4h_sma200 and close[i] < ema50[i] and volume_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA50
            if close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price crosses above EMA50
            if close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Structure_Trend_Filter"
timeframe = "1h"
leverage = 1.0
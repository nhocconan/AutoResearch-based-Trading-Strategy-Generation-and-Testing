#!/usr/bin/env python3
# 6h_cci_trend_volume_v1
# Hypothesis: Combines CCI for trend strength with volume confirmation and daily trend filter.
# In bull markets: CCI > 100 + volume > avg + price > daily EMA200 = long
# In bear markets: CCI < -100 + volume > avg + price < daily EMA200 = short
# Uses daily EMA200 for trend filter to avoid counter-trend trades.
# Target: 15-30 trades/year for low fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_cci_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily trend filter (1d EMA200) - load once before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily data
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 6h indicators
    # CCI(20)
    tp = (high + low + close) / 3.0
    sma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(tp).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (tp - sma_tp) / (0.015 * mad)
    
    # Volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 40  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(cci[i]) or np.isnan(avg_volume[i]) or np.isnan(ema200_1d_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Daily trend filter
        daily_uptrend = close[i] > ema200_1d_aligned[i]
        daily_downtrend = close[i] < ema200_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.2 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: CCI drops below 0 or trend changes
            if cci[i] < 0 or not daily_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI rises above 0 or trend changes
            if cci[i] > 0 or not daily_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_ok:
                # Long entry: CCI > 100 + uptrend
                if cci[i] > 100 and daily_uptrend:
                    position = 1
                    signals[i] = 0.25
                # Short entry: CCI < -100 + downtrend
                elif cci[i] < -100 and daily_downtrend:
                    position = -1
                    signals[i] = -0.25
    
    return signals
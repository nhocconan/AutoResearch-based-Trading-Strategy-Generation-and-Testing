#!/usr/bin/env python3
# 1h_4h_1d_volume_trend_follow_v1
# Hypothesis: 1h trend following with 4h/1d filters. Long when 4h MA > price and 1d bullish, short when 4h MA < price and 1d bearish.
# Uses volume confirmation to avoid false breaks. Session filter (08-20 UTC) reduces noise.
# Target: 15-30 trades/year per symbol. Works in bull via 4h uptrend, bear via 4h downtrend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_volume_trend_follow_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h EMA21 for trend
    close_s = pd.Series(close)
    ema21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 4h close for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # 4h EMA50
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # 1d EMA100
    ema100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # Volume ratio (current vs 20-period average)
    vol_s = pd.Series(volume)
    vol_ma20 = vol_s.rolling(window=20, min_periods=20).mean().values
    vol_ratio = vol_s.values / np.where(vol_ma20 > 0, vol_ma20, 1)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(50, 21)
    
    for i in range(start_idx, n):
        # Skip if session outside 08-20 UTC
        if not (8 <= hours[i] <= 20):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema21[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(ema100_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: require above average volume
        vol_ok = vol_ratio[i] > 1.2
        
        if position == 1:  # Long position
            # Exit: price crosses below EMA21 or 4h trend turns bearish
            if close[i] < ema21[i] or ema50_4h_aligned[i] < ema100_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price crosses above EMA21 or 4h trend turns bullish
            if close[i] > ema21[i] or ema50_4h_aligned[i] > ema100_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long: price above EMA21, 4h bullish vs 1d, volume confirmation
            if (close[i] > ema21[i] and 
                ema50_4h_aligned[i] > ema100_1d_aligned[i] and 
                vol_ok):
                position = 1
                signals[i] = 0.20
            # Short: price below EMA21, 4h bearish vs 1d, volume confirmation
            elif (close[i] < ema21[i] and 
                  ema50_4h_aligned[i] < ema100_1d_aligned[i] and 
                  vol_ok):
                position = -1
                signals[i] = -0.20
    
    return signals
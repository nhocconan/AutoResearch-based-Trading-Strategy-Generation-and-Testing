#!/usr/bin/env python3
"""
4h_volume_accumulation_distribution_12h_trend_v1
Hypothesis: Accumulation/Distribution (AD) line confirms institutional accumulation/distribution.
Price breaking above/below 4h swing high/low with AD confirmation and 12h trend alignment
captures strong moves. Works in bull markets (accumulation/distribution continues) and
bear markets (distribution/accumulation continues) by trading with the 12h trend.
Target: 20-50 trades/year by requiring swing break + AD divergence + trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_volume_accumulation_distribution_12h_trend_v1"
timeframe = "4h"
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
    
    # Accumulation/Distribution Line
    clv = ((close - low) - (high - close)) / (high - low)
    clv = np.where((high - low) == 0, 0, clv)
    adl = np.cumsum(clv * volume)
    
    # 4h swing points (10-period lookback)
    lookback = 10
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        window_high = high[i-lookback:i+1]
        window_low = low[i-lookback:i+1]
        swing_high[i] = np.max(window_high)
        swing_low[i] = np.min(window_low)
    
    # 12h trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # AD trend (20-period slope)
    ad_sma = pd.Series(adl).rolling(window=20, min_periods=20).mean().values
    ad_slope = np.diff(ad_sma, prepend=ad_sma[0])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(swing_high[i]) or 
            np.isnan(swing_low[i]) or 
            np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(ad_slope[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below swing low OR AD turns down
            if close[i] < swing_low[i] or ad_slope[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above swing high OR AD turns up
            if close[i] > swing_high[i] or ad_slope[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above swing high + AD rising + uptrend
            if (close[i] > swing_high[i] and 
                ad_slope[i] > 0 and 
                close[i] > ema50_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below swing low + AD falling + downtrend
            elif (close[i] < swing_low[i] and 
                  ad_slope[i] < 0 and 
                  close[i] < ema50_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
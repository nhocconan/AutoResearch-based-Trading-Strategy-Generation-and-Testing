#!/usr/bin/env python3
"""
1H 4H Trend + Volume Spike + Price Channel Breakout
Hypothesis: In trending markets (identified by 4h EMA alignment), 1h breakouts of 20-period price channels
with volume spikes capture momentum moves. Works in bull (uptrend breakouts) and bear (downtrend breakdowns).
Uses volume confirmation to avoid false breakouts. Low trade frequency via trend filter and volume spike requirement.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data once for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h EMA20 and EMA50 for trend direction
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Trend: 1 = uptrend (EMA20 > EMA50), -1 = downtrend (EMA20 < EMA50), 0 = no trend
    trend_4h = np.where(ema20_4h > ema50_4h, 1, np.where(ema20_4h < ema50_4h, -1, 0))
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # 1h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h price channels: highest high and lowest low of last 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if (np.isnan(trend_4h_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        if position == 0:
            # Look for breakouts in direction of 4h trend
            if trend_4h_aligned[i] == 1:  # Uptrend
                # Long breakout: price breaks above 20-period high with volume spike
                if price > highest_high[i] and vol_spike[i]:
                    signals[i] = 0.20
                    position = 1
            elif trend_4h_aligned[i] == -1:  # Downtrend
                # Short breakdown: price breaks below 20-period low with volume spike
                if price < lowest_low[i] and vol_spike[i]:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:
            # Long exit: price breaks below 20-period low or trend changes
            if price < lowest_low[i] or trend_4h_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price breaks above 20-period high or trend changes
            if price > highest_high[i] or trend_4h_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1H_4HTrend_VolumeSpike_ChannelBreakout"
timeframe = "1h"
leverage = 1.0
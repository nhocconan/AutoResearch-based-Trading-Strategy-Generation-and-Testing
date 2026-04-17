#!/usr/bin/env python3
"""
1h_VolumeBreakout_4hTrend
Hypothesis: On 1h, buy breakouts above the 4h high of the last 20 periods with volume confirmation, sell breakdowns below the 4h low with volume. Uses 4h structure for direction, 1h for precise entry. Volume filter ensures conviction. Designed for 15-30 trades/year to avoid fee drag in choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data for structure (high/low) and trend ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h highest high of last 20 periods (for breakout level)
    hh20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # 4h lowest low of last 20 periods (for breakdown level)
    ll20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h levels to 1h timeframe (already delayed for completed bar)
    hh20_4h_aligned = align_htf_to_ltf(prices, df_4h, hh20_4h)
    ll20_4h_aligned = align_htf_to_ltf(prices, df_4h, ll20_4h)
    
    # 4h trend: EMA34 of close
    ema34_4h = pd.Series(close_4h).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Volume filter: 1h volume > 1.5x 20-period average
    vol_avg20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: covers 4h indicators
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(hh20_4h_aligned[i]) or 
            np.isnan(ll20_4h_aligned[i]) or 
            np.isnan(ema34_4h_aligned[i]) or 
            np.isnan(vol_avg20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_filter = volume[i] > 1.5 * vol_avg20[i]
        
        # Entry conditions
        if position == 0:
            # Long: break above 4h HH20 + above 4h EMA34 (uptrend) + volume
            if close[i] > hh20_4h_aligned[i] and close[i] > ema34_4h_aligned[i] and vol_filter:
                signals[i] = 0.20
                position = 1
                continue
            # Short: break below 4h LL20 + below 4h EMA34 (downtrend) + volume
            elif close[i] < ll20_4h_aligned[i] and close[i] < ema34_4h_aligned[i] and vol_filter:
                signals[i] = -0.20
                position = -1
                continue
        
        # Exit conditions: reverse on opposite break
        elif position == 1:
            if close[i] < ll20_4h_aligned[i]:  # reverse to short on breakdown
                signals[i] = -0.20
                position = -1
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            if close[i] > hh20_4h_aligned[i]:  # reverse to long on breakout
                signals[i] = 0.20
                position = 1
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_VolumeBreakout_4hTrend"
timeframe = "1h"
leverage = 1.0
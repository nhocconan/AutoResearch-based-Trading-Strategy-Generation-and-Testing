#!/usr/bin/env python3
"""
12h Price Channel Breakout with 1w Trend and Volume Filter
Hypothesis: Price breaking above/below 1-week high/low with volume confirmation
and aligned with 1-week trend captures institutional moves. Works in both
bull and bear markets by filtering counter-trend trades with 1w EMA.
Target: 15-30 trades/year to minimize fee decay while capturing strong moves.
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
    
    # Get 1w data for trend and channel (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1w high/low for price channel (50-period lookback)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate rolling max/min for 1w channel
    high_max = pd.Series(high_1w).rolling(window=50, min_periods=50).max().values
    low_min = pd.Series(low_1w).rolling(window=50, min_periods=50).min().values
    
    # Align to 12h timeframe (wait for 1w bar to close)
    high_max_aligned = align_htf_to_ltf(prices, df_1w, high_max)
    low_min_aligned = align_htf_to_ltf(prices, df_1w, low_min)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current volume > 2.0x 30-period volume average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(high_max_aligned[i]) or 
            np.isnan(low_min_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        trend = ema50_1w_aligned[i]
        channel_high = high_max_aligned[i]
        channel_low = low_min_aligned[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: price breaks above 1w high with volume, in uptrend
            if price > channel_high and vol_ok and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w low with volume, in downtrend
            elif price < channel_low and vol_ok and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns below 1w high or trend weakens
            if price < channel_high or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns above 1w low or trend weakens
            if price > channel_low or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_PriceChannel_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0
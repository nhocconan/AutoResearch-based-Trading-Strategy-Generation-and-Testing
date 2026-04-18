#!/usr/bin/env python3
"""
12h Donchian Breakout with 1d Trend Filter
Hypothesis: Price breaks above/below Donchian channels on 12h timeframe with volume confirmation
and 1d trend alignment capture sustained moves in both bull and bear markets. The 1d trend filter
avoids counter-trend trades while volume ensures momentum validity. This strategy targets
15-25 trades/year to minimize fee decay.
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
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian channels on 12h
    period = 20
    high_12h = get_htf_data(prices, '12h')['high'].values
    low_12h = get_htf_data(prices, '12h')['low'].values
    
    # 12h Donchian upper/lower
    donch_high = pd.Series(high_12h).rolling(window=period, min_periods=period).max().values
    donch_low = pd.Series(low_12h).rolling(window=period, min_periods=period).min().values
    
    # Align Donchian levels to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '12h'), donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '12h'), donch_low)
    
    # Volume filter: current volume > 1.8x 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        trend = ema50_1d_aligned[i]
        vol_ok = vol_filter[i]
        upper = donch_high_aligned[i]
        lower = donch_low_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume, in uptrend
            if price > upper and vol_ok and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume, in downtrend
            elif price < lower and vol_ok and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns to Donchian mid or trend weakens
            mid = (upper + lower) / 2
            if price < mid or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to Donchian mid or trend weakens
            mid = (upper + lower) / 2
            if price > mid or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0
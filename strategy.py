#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Confirmation and 1d Trend Filter
Hypothesis: Price breaking out of 20-period Donchian channels with above-average volume
and aligned with 1d EMA trend captures strong directional moves. Works in both bull and
bear markets by filtering breakouts with higher timeframe trend. Low trade frequency
(~25-35/year) minimizes fee decay while capturing explosive moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Donchian Channel (20-period high/low)
    donch_period = 20
    upper = pd.Series(high).rolling(window=donch_period, min_periods=donch_period).max().values
    lower = pd.Series(low).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # Volume filter: current volume > 1.5x 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        trend = ema34_1d_aligned[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Look for Donchian breakout with volume, in trend direction
            if vol_ok:
                # Breakout above upper channel with volume in uptrend
                if price > upper[i] and price > trend:
                    signals[i] = 0.25
                    position = 1
                # Breakout below lower channel with volume in downtrend
                elif price < lower[i] and price < trend:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit if price returns to opposite Donchian band
            if price < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to opposite Donchian band
            if price > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_1dTrend"
timeframe = "4h"
leverage = 1.0
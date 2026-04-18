#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout with Volume Confirmation and 1w Trend Filter
Hypothesis: Daily Donchian breakouts with volume confirmation and weekly trend alignment
capture sustained moves in both bull and bear markets. The weekly timeframe filters
out counter-trend noise, while volume ensures institutional participation. This
low-frequency approach minimizes fee drag while capturing major trends.
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
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate 20-period Donchian channels on daily
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_ok = vol_filter[i]
        trend = ema34_1w_aligned[i]
        
        if position == 0:
            # Long: break above upper Donchian with volume, in weekly uptrend
            if price > upper and vol_ok and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume, in weekly downtrend
            elif price < lower and vol_ok and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns to midpoint or trend weakens
            midpoint = (upper + lower) / 2
            if price < midpoint or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to midpoint or trend weakens
            midpoint = (upper + lower) / 2
            if price > midpoint or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0
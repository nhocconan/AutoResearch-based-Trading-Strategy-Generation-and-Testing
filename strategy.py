#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Confirmation and 12h Trend Filter
Hypothesis: Donchian channel breakouts capture institutional momentum moves.
In both bull and bear markets, breakouts with volume confirmation and trend alignment
provide edge. Using 12h EMA as trend filter avoids counter-trend trades.
Targets 20-30 trades/year to minimize fee drag while capturing strong momentum.
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
    
    # Get 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # 12h EMA34 for trend filter
    ema34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Donchian channels (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = vol_filter[i]
        trend = ema34_12h_aligned[i]
        
        if position == 0:
            # Long: break above Donchian high with volume, in uptrend
            if price > donchian_high[i] and vol_ok and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume, in downtrend
            elif price < donchian_low[i] and vol_ok and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns to Donchian midpoint or trend weakens
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if price < midpoint or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to Donchian midpoint or trend weakens
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if price > midpoint or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_12hTrend"
timeframe = "4h"
leverage = 1.0
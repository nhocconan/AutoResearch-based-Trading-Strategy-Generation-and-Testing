#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Spike and 1d Trend Filter
Hypothesis: Donchian channel breakouts (20-period) capture momentum. 
Trading only in direction of 1d EMA50 trend filters counter-trend moves.
Volume spike (>2x 20-period average) confirms breakout strength.
Low frequency (~20-40/year) minimizes fee drag while capturing strong trends.
Works in bull/bear by only taking breakouts aligned with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 2x 20-period volume average (on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = high_roll[i]
        lower = low_roll[i]
        trend = ema50_1d_aligned[i]
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Look for Donchian breakout with volume, in trend direction
            if vol_ok:
                # Break above upper band in uptrend
                if price > upper and price > trend:
                    signals[i] = 0.25
                    position = 1
                # Break below lower band in downtrend
                elif price < lower and price < trend:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit if price returns to lower band or trend reverses
            if price < lower or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to upper band or trend reverses
            if price > upper or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0
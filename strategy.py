#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout with Volume Confirmation and Weekly Trend Filter
Hypothesis: Weekly Donchian(20) channel breakouts on the daily timeframe capture
institutional breakout moves. Volume confirmation ensures genuine institutional
participation, while the weekly EMA filter ensures alignment with the higher
timeframe trend, working in both bull and bear markets by filtering counter-trend
trades. Target: 10-25 trades/year to minimize fee drift while capturing strong moves.
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
    
    # 20-period EMA on weekly close for trend filter
    ema20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate daily Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-day volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for Donchian
    
    for i in range(start_idx, n):
        if np.isnan(ema20_1w_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        trend = ema20_1w_aligned[i]
        upper = high_20[i]
        lower = low_20[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band with volume, in weekly uptrend
            if price > upper and vol_ok and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band with volume, in weekly downtrend
            elif price < lower and vol_ok and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns below upper band or weekly trend weakens
            if price < upper or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns above lower band or weekly trend weakens
            if price > lower or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0
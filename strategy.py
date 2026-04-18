#!/usr/bin/env python3
"""
4h Donchian Breakout with 1d Trend Filter and Volume Confirmation
Hypothesis: Price breaking out of Donchian channels (20-period) with volume confirmation
and aligned with 1d EMA trend captures strong directional moves. Trend filter avoids
counter-trend trades, reducing whipsaw in sideways markets. Low trade frequency
(~20-30/year) minimizes fee drag while capturing explosive moves in both bull and bear
markets by following the higher timeframe trend.
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
    donchian_period = 20
    upper = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume filter: current volume > 1.5x 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ok = volume > (vol_ma * 1.5)
    
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
        vol = vol_ok[i]
        
        if position == 0:
            # Look for breakout with volume in trend direction
            if vol:
                # Bullish breakout: price breaks above upper Donchian band in uptrend
                if price > upper[i] and price > trend:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below lower Donchian band in downtrend
                elif price < lower[i] and price < trend:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit if price returns to the Donchian midpoint or breaks below lower band
            midpoint = (upper[i] + lower[i]) / 2
            if price < midpoint or price < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to the Donchian midpoint or breaks above upper band
            midpoint = (upper[i] + lower[i]) / 2
            if price > midpoint or price > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0
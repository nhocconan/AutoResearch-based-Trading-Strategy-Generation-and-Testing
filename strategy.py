#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 12h EMA trend filter and volume spike.
# Williams %R measures momentum: -20 to 0 = overbought, -80 to -100 = oversold.
# Strategy: In trending markets (12h EMA slope positive/negative), buy oversold dips in uptrend, sell overbought rallies in downtrend.
# Volume spike confirms institutional participation. Designed for ~20-30 trades/year per symbol.
# Works in both bull (buy dips) and bear (sell rallies) by following higher timeframe trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R calculation (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    diff = highest_high - lowest_low
    diff = np.where(diff == 0, 1e-10, diff)
    williams_r = -100 * (highest_high - close) / diff
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 50-period EMA on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_slope = np.diff(ema50_12h, prepend=ema50_12h[0])  # slope of EMA
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    ema50_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h_slope)
    
    # Volume filter: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(ema50_12h_slope_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R levels
        oversold = -80
        overbought = -20
        
        # Trend filter: 12h EMA slope
        uptrend = ema50_12h_slope_aligned[i] > 0
        downtrend = ema50_12h_slope_aligned[i] < 0
        
        # Entry logic
        if uptrend and williams_r[i] < oversold and volume_filter[i]:
            # Buy oversold dip in uptrend
            signals[i] = 0.25
            position = 1
        elif downtrend and williams_r[i] > overbought and volume_filter[i]:
            # Sell overbought rally in downtrend
            signals[i] = -0.25
            position = -1
        else:
            # Hold position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
                
            # Exit conditions: Williams %R returns to neutral zone (-50)
            if position == 1 and williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            elif position == -1 and williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_WilliamsR_12hEMA50_VolumeFilter"
timeframe = "4h"
leverage = 1.0
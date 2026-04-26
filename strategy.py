#!/usr/bin/env python3
"""
6h_ElderRay_Breakout_12hTrend_Volume
Hypothesis: 6h Elder Ray (Bull/Bear Power) breakout with 12h trend filter (EMA50) and volume spike.
Enters long when Bull Power > 0 (close > EMA13) AND price breaks above prior 6h high with volume spike AND 12h EMA50 uptrend.
Enters short when Bear Power < 0 (close < EMA13) AND price breaks below prior 6h low with volume spike AND 12h EMA50 downtrend.
Exits when opposing Elder Ray power crosses zero.
Designed for 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
Uses discrete position sizing (0.25) to minimize churn. Works in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray (primary timeframe)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = close - ema13  # >0 = bullish
    bear_power = close - ema13  # <0 = bearish (same calc, check sign)
    
    # Prior 6h high/low for breakout (shifted by 1 to avoid look-ahead)
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    # Load 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need EMA13 + prior bar)
    start_idx = 13 + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(prior_high[i]) or 
            np.isnan(prior_low[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: Bull Power > 0 + break above prior high + volume spike + 12h uptrend
        if bull_power[i] > 0 and close[i] > prior_high[i] and volume_spike[i] and close[i] > ema_50_12h_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: Bear Power < 0 + break below prior low + volume spike + 12h downtrend
        elif bear_power[i] < 0 and close[i] < prior_low[i] and volume_spike[i] and close[i] < ema_50_12h_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: opposing Elder Ray power crosses zero (long exits when Bear Power >=0, short exits when Bull Power <=0)
        elif position == 1 and bear_power[i] >= 0:
            signals[i] = 0.0
            position = 0
        elif position == -1 and bull_power[i] <= 0:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_ElderRay_Breakout_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0
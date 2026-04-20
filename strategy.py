#!/usr/bin/env python3
"""
12h_WVWAP_Trend_With_Weekly_Filter
Hypothesis: Trade 12h VWAP with weekly trend filter and volume confirmation. 
Long when price > VWAP + weekly uptrend + volume spike; short when price < VWAP + weekly downtrend + volume spike.
VWAP captures institutional sentiment, weekly filter avoids counter-trend trades, volume spike confirms institutional participation.
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
Works in bull/bear: weekly trend filter avoids counter-trend trades, volume filter reduces false signals.
"""

name = "12h_WVWAP_Trend_With_Weekly_Filter"
timeframe = "12h"
leverage = 1.0

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
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_weekly = df_weekly['close'].values
    ema20_weekly = np.full_like(close_weekly, np.nan)
    if len(close_weekly) >= 20:
        multiplier = 2.0 / (20 + 1)
        ema20_weekly[19] = np.mean(close_weekly[:20])
        for i in range(20, len(close_weekly)):
            ema20_weekly[i] = multiplier * close_weekly[i] + (1 - multiplier) * ema20_weekly[i-1]
    ema20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    
    # Calculate VWAP (typical price * volume cumulative)
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.nancumsum(pv)
    cum_vol = np.nancumsum(volume)
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    
    # Calculate volume filter (volume > 1.5x 20-period average)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, n):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap[i]) or np.isnan(ema20_weekly_aligned[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > VWAP + weekly uptrend + volume spike
            if close[i] > vwap[i] and close[i] > ema20_weekly_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < VWAP + weekly downtrend + volume spike
            elif close[i] < vwap[i] and close[i] < ema20_weekly_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < VWAP OR weekly trend turns down
            if close[i] < vwap[i] or close[i] < ema20_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > VWAP OR weekly trend turns up
            if close[i] > vwap[i] or close[i] > ema20_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
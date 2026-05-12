#!/usr/bin/env python3
# 12H_MARKET_PROFILE_POINT_OF_CONTROL_1W_TREND_FILTER
# Hypothesis: Uses the weekly Point of Control (POC) from Market Profile as a key support/resistance level.
# Price is expected to revert to the POC after deviation, with the weekly trend (EMA34) filtering direction.
# Works in both bull and bear markets: In uptrend, go long when price deviates below POC; in downtrend, go short when price deviates above POC.
# Volume-weighted POC identifies institutional value areas, and trend alignment reduces counter-trend whipsaws.
# Target: 15-25 trades/year on 12h timeframe.

name = "12H_MARKET_PROFILE_POINT_OF_CONTROL_1W_TREND_FILTER"
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
    
    # Weekly data for Point of Control (POC) calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly typical price and volume for POC (simplified as volume-weighted average price)
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vp = typical_price * df_1w['volume']  # Volume * typical price
    # POC = price level with highest volume (approximated by VWAP for stability)
    poc = (vp.cumsum() / df_1w['volume'].cumsum()).values
    
    # Weekly EMA for trend filter (34-period)
    ema34 = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align POC and EMA to 12h timeframe
    poc_aligned = align_htf_to_ltf(prices, df_1w, poc)
    ema34_aligned = align_htf_to_ltf(prices, df_1w, ema34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 10  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(poc_aligned[i]) or np.isnan(ema34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price deviates below POC in uptrend (mean reversion to value area)
            if (close[i] < poc_aligned[i] and close[i] > ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price deviates above POC in downtrend (mean reversion to value area)
            elif (close[i] > poc_aligned[i] and close[i] < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to or exceeds POC
            if close[i] >= poc_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to or goes below POC
            if close[i] <= poc_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
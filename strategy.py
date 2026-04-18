#!/usr/bin/env python3
"""
6h Elder Ray Power with 1d Trend and Volume Filter
Hypothesis: Elder Ray Bull/Bear Power identifies institutional buying/selling pressure.
Combining with 1d EMA trend filter and volume confirmation captures sustained moves.
Works in bull markets (Bull Power > 0 + uptrend) and bear markets (Bear Power < 0 + downtrend).
Targets 20-30 trades/year to minimize fee drag while capturing institutional flows.
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
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA20 for trend filter
    ema20_1d = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate Elder Ray components
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume filter: current volume > 1.5x 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        trend = ema20_1d_aligned[i]
        bp = bull_power[i]
        br = bear_power[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: Bull Power > 0 (buying pressure) + volume + uptrend
            if bp > 0 and vol_ok and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (selling pressure) + volume + downtrend
            elif br < 0 and vol_ok and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if Bull Power turns negative or trend weakens
            if bp <= 0 or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if Bear Power turns positive or trend weakens
            if br >= 0 or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0
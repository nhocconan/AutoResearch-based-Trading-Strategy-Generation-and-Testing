# US Patent 7103525 - Market Timing System  
# Hypothesis: US Patent 7103525-based cycle indicator combined with weekly trend filter on 6h timeframe  
# The patented system identifies trend exhaustion using a mathematical formula (13-period EMA of high-low differential)  
# Weekly trend filter prevents counter-trend trades in strong trends, improving win rate in both bull and bear markets  
# Target: 50-150 trades over 4 years (12-37/year) with position size 0.25  

#!/usr/bin/env python3
name = "6h_US_Patent_7103525_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Weekly EMA21 for trend filter (from patent reference)
    ema_21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_6h = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # US Patent 7103525 Market Timing System calculation
    # Core formula: 13-period EMA of (High - Low) differential
    hl_diff = high - low
    # Apply 13-period EMA to the high-low differential
    ema_13_hl = pd.Series(hl_diff).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Signal generation logic from the patent
    # Long signal: When current close is above (previous close + EMA13_HL_diff)
    # Short signal: When current close is below (previous close - EMA13_HL_diff)
    signal_long_raw = close > np.roll(close, 1) + ema_13_hl
    signal_short_raw = close < np.roll(close, 1) - ema_13_hl
    
    # Handle first element (no previous close)
    signal_long_raw[0] = False
    signal_short_raw[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 13)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_21_6h[i]) or np.isnan(ema_13_hl[i]) or 
            np.isnan(signal_long_raw[i]) or np.isnan(signal_short_raw[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter: only take longs in weekly uptrend, shorts in weekly downtrend
        weekly_uptrend = close[i] > ema_21_6h[i]
        weekly_downtrend = close[i] < ema_21_6h[i]
        
        if position == 0:
            # Long: patent long signal + weekly uptrend
            if signal_long_raw[i] and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: patent short signal + weekly downtrend
            elif signal_short_raw[i] and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: patent exit signal or trend reversal
            if signal_short_raw[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: patent exit signal or trend reversal
            if signal_long_raw[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
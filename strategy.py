#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray with weekly trend filter and volume confirmation
# Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Long when Bull Power > 0, weekly EMA21 uptrend, and volume spike
# Short when Bear Power > 0, weekly EMA21 downtrend, and volume spike
# Elder Ray measures bull/bear power relative to EMA, effective in trending markets
# Weekly EMA filters for higher timeframe trend alignment
# Volume spike confirms institutional participation
# Targets 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

name = "6h_ElderRay_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate weekly EMA(21) for trend filter
    weekly_close = df_1w['close'].values
    ema21_1w = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Calculate EMA13 for Elder Ray (13-period EMA of close)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = ema13 - low   # Bear Power = EMA13 - Low
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema21_1w_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema21_1w_val = ema21_1w_aligned[i]
        bp = bull_power[i]
        br = bear_power[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Bull Power > 0, weekly uptrend, volume spike
            if bp > 0 and ema21_1w_val > ema21_1w_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power > 0, weekly downtrend, volume spike
            elif br > 0 and ema21_1w_val < ema21_1w_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 or weekly trend turns down
            if bp <= 0 or ema21_1w_val < ema21_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power <= 0 or weekly trend turns up
            if br <= 0 or ema21_1w_val > ema21_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
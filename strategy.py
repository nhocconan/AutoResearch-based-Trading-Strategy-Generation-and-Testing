#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index with daily trend filter and volume confirmation
# Long when Bull Power crosses above zero (bullish momentum), daily trend up, volume spike
# Short when Bear Power crosses above zero (bearish weakening), daily trend down, volume spike
# Elder Ray measures bull/bear power relative to EMA; daily trend filters for higher timeframe direction
# Volume spike confirms institutional participation; avoids false signals
# Targets 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

name = "6h_ElderRay_DailyTrend_Volume"
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
    
    # Get daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(13) for Elder Ray and trend filter
    daily_close = df_1d['close'].values
    ema13_1d = pd.Series(daily_close).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Calculate Elder Ray components (13-period)
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema13_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema13_1d_val = ema13_1d_aligned[i]
        bp = bull_power[i]
        br = bear_power[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Bull Power crosses above zero (bullish momentum), daily uptrend, volume spike
            if bp > 0 and bull_power[i-1] <= 0 and ema13_1d_val > 0 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power crosses above zero (bearish weakening), daily downtrend, volume spike
            elif br > 0 and bear_power[i-1] <= 0 and ema13_1d_val < 0 and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power falls below zero or daily trend turns down
            if bp <= 0 or ema13_1d_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power falls below zero or daily trend turns up
            if br <= 0 or ema13_1d_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
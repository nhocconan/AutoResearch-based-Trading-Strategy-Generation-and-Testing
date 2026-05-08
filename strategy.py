#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour SuperTrend with weekly trend filter and volume confirmation
# Long when price > SuperTrend and weekly EMA(34) uptrend and volume spike
# Short when price < SuperTrend and weekly EMA(34) downtrend and volume spike
# SuperTrend adapts to volatility; weekly EMA provides higher timeframe bias
# Volume spike confirms institutional participation; avoids choppy false breakouts
# Targets 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

name = "6h_SuperTrend_WeeklyTrend_Volume"
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
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    weekly_close = df_1w['close'].values
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate SuperTrend
    # ATR calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Basic upper and lower bands
    basic_upper = (high + low) / 2 + 3.0 * atr
    basic_lower = (high + low) / 2 - 3.0 * atr
    
    # Final upper and lower bands
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    
    for i in range(1, n):
        if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i-1]
            
        if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i-1]
    
    # SuperTrend
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 for uptrend, -1 for downtrend
    supertrend[0] = final_lower[0]
    direction[0] = 1
    
    for i in range(1, n):
        if close[i] > final_upper[i-1]:
            direction[i] = 1
        elif close[i] < final_lower[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            supertrend[i] = final_lower[i]
        else:
            supertrend[i] = final_upper[i]
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(supertrend[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1w_val = ema34_1w_aligned[i]
        price = close[i]
        st = supertrend[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price > SuperTrend and weekly uptrend and volume spike
            if price > st and ema34_1w_val > ema34_1w_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: price < SuperTrend and weekly downtrend and volume spike
            elif price < st and ema34_1w_val < ema34_1w_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < SuperTrend or weekly trend turns down
            if price < st or ema34_1w_val < ema34_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > SuperTrend or weekly trend turns up
            if price > st or ema34_1w_val > ema34_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray combination with 1d trend filter
# Uses Williams Alligator (Jaw/Teeth/Lips) to identify trend absence/presence
# Elder Ray (Bull/Bear Power) measures trend strength relative to EMA13
# 1d EMA50 ensures alignment with medium-term trend to avoid counter-trend trades
# Designed for 50-150 total trades over 4 years (12-37/year) on 12h timeframe
# Works in trending markets (Alligator awake) and avoids chop (Alligator sleeping)
# BTC and ETH focused with SOL as validation

name = "12h_WilliamsAlligator_ElderRay_1dEMA50"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) - all SMMA
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA*(period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Alligator lines)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Alligator awakening: Lips > Teeth > Jaw (bullish) OR Lips < Teeth < Jaw (bearish)
            # Elder Ray confirmation: Bull Power > 0 (bullish) or Bear Power < 0 (bearish)
            # 1d trend filter: price > EMA50 (bullish) or price < EMA50 (bearish)
            
            # Long entry: Alligator bullish alignment + Bull Power > 0 + price > 1d EMA50
            if (lips[i] > teeth[i] > jaw[i] and 
                bull_power[i] > 0 and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Alligator bearish alignment + Bear Power < 0 + price < 1d EMA50
            elif (lips[i] < teeth[i] < jaw[i] and 
                  bear_power[i] < 0 and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator sleeping (Lips < Teeth OR Teeth < Jaw) OR trend change
            if lips[i] < teeth[i] or teeth[i] < jaw[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator sleeping (Lips > Teeth OR Teeth > Jaw) OR trend change
            if lips[i] > teeth[i] or teeth[i] > jaw[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
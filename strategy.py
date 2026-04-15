#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1w Trend Filter + Volume Confirmation
# Uses Elder Ray (Bull/Bear Power) to measure buying/selling pressure,
# weekly trend (price vs 40-week EMA) to filter direction,
# and volume spike to confirm institutional participation.
# Works in bull markets via long signals when bull power > 0 and price above weekly EMA.
# Works in bear markets via short signals when bear power < 0 and price below weekly EMA.
# Target: 60-120 total trades over 4 years (15-30/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data (primary timeframe) for price action and Elder Ray
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate EMA13 and EMA20 on 6h for Elder Ray
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema20_6h = pd.Series(close_6h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate EMA40 on weekly for trend filter
    ema40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    
    # Calculate Elder Ray components
    bull_power = high_6h - ema13_6h  # Bull Power: High - EMA13
    bear_power = low_6h - ema20_6h   # Bear Power: Low - EMA20
    
    # Volume average (20-period on 6h)
    vol_avg_6h = pd.Series(df_6h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    ema13_6h_aligned = align_htf_to_ltf(prices, df_6h, ema13_6h)
    ema20_6h_aligned = align_htf_to_ltf(prices, df_6h, ema20_6h)
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    vol_avg_aligned = align_htf_to_ltf(prices, df_6h, vol_avg_6h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema13_6h_aligned[i]) or np.isnan(ema20_6h_aligned[i]) or
            np.isnan(ema40_1w_aligned[i]) or np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: bull power > 0 (buying pressure) + price above weekly EMA40 + volume spike
        if (bull_power_aligned[i] > 0 and
            close[i] > ema40_1w_aligned[i] and
            volume[i] > 1.8 * vol_avg_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: bear power < 0 (selling pressure) + price below weekly EMA40 + volume spike
        elif (bear_power_aligned[i] < 0 and
              close[i] < ema40_1w_aligned[i] and
              volume[i] > 1.8 * vol_avg_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or loss of momentum (power crosses zero)
        elif position == 1 and (bull_power_aligned[i] <= 0 or close[i] < ema40_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bear_power_aligned[i] >= 0 or close[i] > ema40_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0
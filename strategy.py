#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Spike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for Camarilla, trend, and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Using (H+L+C)/3 as pivot, standard Camarilla multipliers
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Resistance 1 and Support 1 levels
    r1 = pivot + (range_hl * 1.1 / 12)
    s1 = pivot - (range_hl * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily volume average (20-period)
    vol_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Volume spike condition: current volume > 1.5x daily average
            vol_spike = volume[i] > (vol_avg_aligned[i] * 1.5)
            
            # Long: price breaks above R1 with volume spike and above daily EMA34
            long_cond = (close[i] > r1_aligned[i]) and vol_spike and (close[i] > ema34_aligned[i])
            
            # Short: price breaks below S1 with volume spike and below daily EMA34
            short_cond = (close[i] < s1_aligned[i]) and vol_spike and (close[i] < ema34_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Camarilla body (between S1 and R1) OR trend changes
            if (close[i] < r1_aligned[i] and close[i] > s1_aligned[i]) or (close[i] < ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Camarilla body OR trend changes
            if (close[i] < r1_aligned[i] and close[i] > s1_aligned[i]) or (close[i] > ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R1/S1 levels act as strong intraday support/resistance.
# Long when price breaks above R1 with volume spike and above daily EMA34 trend.
# Short when price breaks below S1 with volume spike and below daily EMA34 trend.
# Exits when price returns to the Camarilla body (between S1 and R1) or trend fails.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue).
# Volume spike filters out false breakouts. Trend filter ensures alignment with higher timeframe.
# Target: 20-50 trades/year to minimize fee decay while capturing strong moves.
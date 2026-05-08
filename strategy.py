#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power with 1-week Trend Filter and Volume Spike
# - Uses daily Bull Power (close - EMA13) and Bear Power (EMA13 - high) to measure bull/bear strength
# - Long when Bull Power > 0 and Bear Power < 0 with 1-week uptrend (price > weekly EMA34)
# - Short when Bear Power > 0 and Bull Power < 0 with 1-week downtrend (price < weekly EMA34)
# - Volume spike confirms institutional participation
# - Works in bull/bear by using 1-week trend filter to avoid counter-trend trades
# - Target: 15-35 trades/year to minimize fee drag on 6h timeframe

name = "6h_ElderRay_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Elder Ray calculation (EMA13)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA13 on daily close
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = close_1d - ema13_1d  # Close - EMA13
    bear_power = ema13_1d - high_1d   # EMA13 - High
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1-week EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (bullish) AND Bear Power < 0 (not bearish) with 1-week uptrend + volume spike
            long_cond = (bull_power_aligned[i] > 0 and 
                        bear_power_aligned[i] < 0 and
                        close[i] > ema34_1w_aligned[i] and
                        volume_spike[i])
            
            # Short: Bear Power > 0 (bearish) AND Bull Power < 0 (not bullish) with 1-week downtrend + volume spike
            short_cond = (bear_power_aligned[i] > 0 and 
                         bull_power_aligned[i] < 0 and
                         close[i] < ema34_1w_aligned[i] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 (loss of bullish momentum) OR Bear Power >= 0 (bearish pressure)
            if bull_power_aligned[i] <= 0 or bear_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power <= 0 (loss of bearish momentum) OR Bull Power >= 0 (bullish pressure)
            if bear_power_aligned[i] <= 0 or bull_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
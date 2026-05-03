#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation.
# Long when Bull Power > 0 AND Bear Power < 0 AND price above 1d EMA50 with volume spike.
# Short when Bear Power > 0 AND Bull Power < 0 AND price below 1d EMA50 with volume spike.
# Uses Elder Ray to measure bull/bear strength relative to EMA13, 1d EMA50 for higher timeframe trend,
# Volume spike confirms participation. Works in both bull and bear markets by only trading with 1d trend.
# Designed for 6h timeframe to minimize fee drag. Target: 50-150 total trades over 4 years.

name = "6h_ElderRay_1dEMA50_VolumeSpike"
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
    
    # Get 6h data for Elder Ray calculation
    df_6h = get_htf_data(prices, '6h')
    
    if len(df_6h) < 14:
        return np.zeros(n)
    
    # Calculate 6h EMA13 for Elder Ray
    close_6h = df_6h['close'].values
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    bull_power = high_6h - ema13_6h
    bear_power = low_6h - ema13_6h
    
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection (20-period volume MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        trend_up = close_val > ema_50_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_50_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND 1d uptrend AND volume spike
            if bull_val > 0 and bear_val < 0 and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 AND Bull Power < 0 AND 1d downtrend AND volume spike
            elif bear_val > 0 and bull_val < 0 and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 OR Bear Power >= 0 OR price breaks below 1d EMA50
            if bull_val <= 0 or bear_val >= 0 or close_val < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power <= 0 OR Bull Power >= 0 OR price breaks above 1d EMA50
            if bear_val <= 0 or bull_val >= 0 or close_val > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
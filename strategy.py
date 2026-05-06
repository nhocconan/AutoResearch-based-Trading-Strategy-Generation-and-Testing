#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 1d EMA50 trend filter and volume confirmation
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 AND Bear Power rising (less negative) AND price > 1d EMA50 AND volume spike
# Short when Bear Power < 0 AND Bull Power falling (less positive) AND price < 1d EMA50 AND volume spike
# Uses 6h timeframe for entries with 1d trend filter to reduce whipsaw in both bull/bear markets
# Discrete sizing 0.25 to limit fee drag; target 80-180 total trades over 4 years (20-45/year)
# Elder Ray measures bull/bear power behind price moves, effective in trending and ranging markets

name = "6h_ElderRay_1dEMA50_VolumeConfirm_v1"
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
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA50 trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6h EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Calculate Elder Ray slope (1-bar change) for confirmation
    bull_power_slope = bull_power - np.roll(bull_power, 1)
    bear_power_slope = bear_power - np.roll(bear_power, 1)
    bull_power_slope[0] = 0
    bear_power_slope[0] = 0
    
    # Calculate volume spike filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(bull_power_slope[i]) or 
            np.isnan(bear_power_slope[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power rising (less negative) AND uptrend AND volume spike
            if (bull_power[i] > 0 and bear_power_slope[i] > 0 and 
                close[i] > ema50_1d_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power falling (less positive) AND downtrend AND volume spike
            elif (bear_power[i] < 0 and bull_power_slope[i] < 0 and 
                  close[i] < ema50_1d_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power turns negative OR Bull Power falls sharply
            if bear_power[i] < 0 or bull_power_slope[i] < -0.5 * np.std(bull_power[max(0, i-20):i+1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power turns positive OR Bear Power rises sharply
            if bull_power[i] > 0 or bear_power_slope[i] > 0.5 * np.std(bear_power[max(0, i-20):i+1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
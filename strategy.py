#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h EMA50 trend filter + volume confirmation
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength
# Enter long when Bull Power > 0 AND Bear Power rising (momentum) AND price > 12h EMA50 AND volume spike
# Enter short when Bear Power > 0 AND Bull Power falling (momentum) AND price < 12h EMA50 AND volume spike
# Exit when Elder Ray power reverses sign or price crosses EMA13
# Discrete sizing 0.25 to limit fee drag; target 50-150 total trades over 4 years (12-37/year)
# Works in bull markets via trend continuation and bear markets via mean reversion at extremes

name = "6h_ElderRay_12hEMA50_VolumeConfirm_v1"
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
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h EMA50 trend filter
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Elder Ray components (using 13-period EMA)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = ema13 - low   # Bear Power: EMA13 - Low
    
    # Calculate volume spike filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 6h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Bull Power > 0 AND Bear Power rising (momentum) AND uptrend AND volume spike
            if (bull_power[i] > 0 and 
                bear_power[i] > bear_power[i-1] and 
                close[i] > ema50_12h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power > 0 AND Bull Power falling (momentum) AND downtrend AND volume spike
            elif (bear_power[i] > 0 and 
                  bull_power[i] < bull_power[i-1] and 
                  close[i] < ema50_12h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power becomes positive (trend weakening) OR price < EMA13
            if bear_power[i] > 0 or close[i] < ema13[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power becomes positive (trend weakening) OR price > EMA13
            if bull_power[i] > 0 or close[i] > ema13[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
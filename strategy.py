#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation
# Long when price breaks above 4h Camarilla R1 AND volume > 1.8x 30-period average AND close > 4h EMA50
# Short when price breaks below 4h Camarilla S1 AND volume > 1.8x 30-period average AND close < 4h EMA50
# Exit when price crosses 4h Camarilla pivot point (mean reversion to 4h equilibrium)
# Uses R1/S1 for tighter breakout signals, targeting 60-150 total trades over 4 years on 1h timeframe
# Designed to work in bull markets (trend continuation via EMA50 filter) and bear markets (mean reversion at pivot)
# Session filter: 08-20 UTC to avoid low-liquidity periods

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate volume spike filter on 1h
    if len(volume) >= 30:
        vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
        volume_filter = volume > (1.8 * vol_ma_30)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 4h data ONCE before loop for Camarilla levels and EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 40:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivot levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    pivot_4h = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    
    # Camarilla levels: R1 = close + range * 1.1/12, S1 = close - range * 1.1/12
    camarilla_r1 = close_4h + (range_4h * 1.1 / 12)
    camarilla_s1 = close_4h - (range_4h * 1.1 / 12)
    camarilla_pivot = pivot_4h  # Use standard pivot as exit level
    
    # Align 4h indicators to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot)
    
    # Calculate 4h EMA50 trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R1 AND volume spike AND above 4h EMA50
            if (close[i] > camarilla_r1_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla S1 AND volume spike AND below 4h EMA50
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema_50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla pivot (mean reversion)
            if close[i] < camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above Camarilla pivot (mean reversion)
            if close[i] > camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
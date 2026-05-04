#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Uses Donchian channel breakouts for clear entry/exit signals with defined risk
# 12h EMA50 provides intermediate trend filter to avoid counter-trend trades in ranging markets
# Volume confirmation (>1.5x 20-period EMA volume) filters false breakouts
# Discrete sizing 0.25 targets 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Works in bull markets (continuation breakouts with uptrend) and bear markets (continuation breakdowns with downtrend)
# 12h trend filter ensures alignment with intermediate-term structure, reducing whipsaws

name = "4h_Donchian20_12hEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough data for EMA50 calculation
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50) trend filter from prior completed 12h bar
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_shifted = np.roll(ema_50_12h, 1)
    ema_50_12h_shifted[0] = np.nan
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Donchian(20) channels from prior completed 4h bar
    # We need to calculate on 4h data but shift to use prior completed bar
    high_4h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_4h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Shift to use prior completed 4h bar (avoid look-ahead)
    high_4h_shifted = np.roll(high_4h, 1)
    low_4h_shifted = np.roll(low_4h, 1)
    high_4h_shifted[0] = np.nan
    low_4h_shifted[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(high_4h_shifted[i]) or 
            np.isnan(low_4h_shifted[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper band AND price > 12h EMA50 AND volume spike
            if close[i] > high_4h_shifted[i] and close[i] > ema_50_12h_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower band AND price < 12h EMA50 AND volume spike
            elif close[i] < low_4h_shifted[i] and close[i] < ema_50_12h_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian lower band OR price crosses below 12h EMA50
            if close[i] < low_4h_shifted[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian upper band OR price crosses above 12h EMA50
            if close[i] > high_4h_shifted[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike (>1.5x 20 EMA volume)
# Uses Camarilla pivot levels from prior completed 1d for structure (R3/S3 = strong support/resistance)
# 1d EMA34 filter ensures we trade in direction of higher timeframe trend (avoids counter-trend whipsaws)
# Volume confirmation ensures breakout has sufficient participation (>1.5x average volume)
# Discrete sizing 0.25 balances risk and return while minimizing fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 12h timeframe
# Works in both bull (breakout continuation) and bear (breakdown continuation) markets
# Focus on BTC/ETH by requiring 1d trend alignment (avoids SOL-only bias, more robust across regimes)

name = "12h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for EMA calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA(34) trend filter from prior completed 1d bar
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_shifted = np.roll(ema_34_1d, 1)
    ema_34_1d_shifted[0] = np.nan
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_shifted)
    
    # Calculate Camarilla pivots from prior completed 1d bar
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low)
    #          S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    daily_range = high_1d - low_1d
    R3 = close_1d + 1.125 * daily_range
    S3 = close_1d - 1.125 * daily_range
    
    # Shift by 1 to use only prior completed 1d bar (no look-ahead)
    R3_shifted = np.roll(R3, 1)
    S3_shifted = np.roll(S3, 1)
    R3_shifted[0] = np.nan
    S3_shifted[0] = np.nan
    
    # Align Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3_shifted)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND price > 1d EMA34 AND volume spike
            if close[i] > R3_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 AND price < 1d EMA34 AND volume spike
            elif close[i] < S3_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to S3 OR price crosses below 1d EMA34
            if close[i] < S3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to R3 OR price crosses above 1d EMA34
            if close[i] > R3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
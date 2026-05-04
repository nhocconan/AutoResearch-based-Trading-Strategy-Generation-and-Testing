#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation (>1.8x 20 EMA volume)
# Uses Camarilla pivot levels (R1/S1) from 4h for institutional breakout structure
# 4h EMA50 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation filters false breakouts (>1.8x average volume) - calibrated for 1h frequency
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods
# Discrete sizing 0.20 balances risk and return while minimizing fee churn
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe
# Works in bull markets (continuation at R1) and bear markets (continuation at S1)
# Focus on BTC/ETH by requiring 4h trend alignment (avoids SOL-only bias)

name = "1h_Camarilla_R1S1_4hEMA50_VolumeConfirm_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) - prices.index is already DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla calculation and EMA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need enough data for EMA50 calculation
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h EMA(50) trend filter from prior completed 4h bar
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_shifted = np.roll(ema_50_4h, 1)
    ema_50_4h_shifted[0] = np.nan
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h_shifted)
    
    # Calculate Camarilla levels (R1, S1) from prior completed 4h bar
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_range = high_4h - low_4h
    r1_level = close_4h + 1.1 * camarilla_range / 12
    s1_level = close_4h - 1.1 * camarilla_range / 12
    
    # Shift to use prior completed 4h bar
    r1_shifted = np.roll(r1_level, 1)
    s1_shifted = np.roll(s1_level, 1)
    r1_shifted[0] = np.nan
    s1_shifted[0] = np.nan
    
    # Align to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1_shifted)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ema_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R1 AND price > 4h EMA50 AND volume spike
            if close[i] > r1_aligned[i] and close[i] > ema_50_4h_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below S1 AND price < 4h EMA50 AND volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema_50_4h_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to S1 OR price crosses below 4h EMA50
            if close[i] < s1_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to R1 OR price crosses above 4h EMA50
            if close[i] > r1_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
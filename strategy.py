#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation (>1.8x 20 EMA volume)
# Uses Camarilla pivot levels from prior completed 1d bar for structure (breakout at R3/S3 = momentum)
# 4h EMA50 filter ensures we trade in direction of higher timeframe trend (avoids counter-trend whipsaws)
# Volume confirmation ensures breakout has sufficient participation (>1.8x average volume)
# Session filter (08-20 UTC) reduces noise trades outside active market hours
# Discrete sizing 0.20 balances risk and return while minimizing fee churn
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe
# Works in both bull (breakout continuation) and bear (breakdown continuation) markets
# Focus on BTC/ETH by requiring 4h trend alignment (avoids SOL-only bias, more robust across regimes)

name = "1h_Camarilla_R3S3_4hEMA50_VolumeConfirm_Session"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 days for pivot calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivots from prior completed 1d bar
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    rng = high_1d - low_1d
    r4 = typical_price + (rng * 1.1 / 2.0)
    r3 = typical_price + (rng * 1.1 / 4.0)
    s3 = typical_price - (rng * 1.1 / 4.0)
    s4 = typical_price - (rng * 1.1 / 2.0)
    
    # Shift by 1 to use only prior completed 1d bar (no look-ahead)
    r4_shifted = np.roll(r4, 1)
    r3_shifted = np.roll(r3, 1)
    s3_shifted = np.roll(s3, 1)
    s4_shifted = np.roll(s4, 1)
    r4_shifted[0] = np.nan
    r3_shifted[0] = np.nan
    s3_shifted[0] = np.nan
    s4_shifted[0] = np.nan
    
    # Align Camarilla levels to 1h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_shifted)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_shifted)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_shifted)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_shifted)
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need enough data for EMA calculation
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA(50) trend filter from prior completed 4h bar
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_shifted = np.roll(ema_50_4h, 1)
    ema_50_4h_shifted[0] = np.nan
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND price > 4h EMA50 AND volume spike
            if close[i] > r3_aligned[i] and close[i] > ema_50_4h_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below S3 AND price < 4h EMA50 AND volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema_50_4h_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to S3 OR price crosses below 4h EMA50
            if close[i] < s3_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to R3 OR price crosses above 4h EMA50
            if close[i] > r3_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
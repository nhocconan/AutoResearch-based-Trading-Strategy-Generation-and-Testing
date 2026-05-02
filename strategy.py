#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA200 trend filter and volume spike confirmation
# Uses 4h EMA200 for medium-term trend direction to avoid false breakouts, 1h for precise entry timing.
# Volume spike confirms breakout strength. Designed for 60-150 total trades over 4 years (15-37/year) on 1h.
# Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend)
# by only taking trades in direction of 4h EMA200. Session filter (08-20 UTC) reduces noise.

name = "1h_Camarilla_R3S3_Breakout_4hEMA200_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h EMA200 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Calculate prior day's Camarilla levels (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prior_high = df_1d['high'].shift(1).values  # prior day's high
    prior_low = df_1d['low'].shift(1).values    # prior day's low
    prior_close = df_1d['close'].shift(1).values # prior day's close
    
    # Calculate Camarilla levels
    R3 = prior_close + (prior_high - prior_low) * 1.1 / 4
    S3 = prior_close - (prior_high - prior_low) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (wait for prior day to complete)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: 2.0x 20-period average (~5 hours)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA200 and prior day)
    start_idx = max(200, 30)  # 200 for EMA200, 30 for prior day data
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Check for NaN values in indicators
        if (np.isnan(ema_200_4h_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R3 with volume spike AND price > 4h EMA200 (bullish trend)
            if (close[i] > R3_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_200_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S3 with volume spike AND price < 4h EMA200 (bearish trend)
            elif (close[i] < S3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_200_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below R3 (failed breakout) OR price below 4h EMA200 (trend change)
            if close[i] < R3_aligned[i] or close[i] < ema_200_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price crosses above S3 (failed breakdown) OR price above 4h EMA200 (trend change)
            if close[i] > S3_aligned[i] or close[i] > ema_200_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals